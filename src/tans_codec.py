#!/usr/bin/env python3
"""
Educational tANS lossless compressor, aligned with adamrt27/tANS_py.

This script follows the classic tabled ANS construction used in Duda's slides
and in the reference repository:
1. Normalize symbol counts to a power-of-two table size L.
2. Spread symbols into a decoding table.
3. Build decoder entries: symbol, number of bits to read, and next state.
4. Build the matching encoder transition table.
5. Encode symbols into a bitstream and decode them from right to left.

The CLI stores a small JSON container so compressed files can be decoded without
keeping the Coder object in memory.
"""

import argparse
import base64
import collections
import heapq
import json
import math

DEFAULT_TABLE_LOG = 12
MIN_TABLE_LOG = 5
MAX_TABLE_LOG = 15

DecodeEntry = collections.namedtuple("DecodeEntry", ["symbol_index", "nb_bits", "new_state"])


class EncodedPayload:
    def __init__(self, table_log, original_size, symbols, frequencies, bitstream):
        self.table_log = table_log
        self.original_size = original_size
        self.symbols = symbols
        self.frequencies = frequencies
        self.bitstream = bitstream

    def to_bytes(self):
        header = {
            "table_log": self.table_log,
            "original_size": self.original_size,
            "symbols": self.symbols,
            "frequencies": self.frequencies,
            "bitstream": base64.b64encode(pack_bits(self.bitstream)).decode("ascii"),
            "bit_count": len(self.bitstream),
        }
        return json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @staticmethod
    def from_bytes(raw_payload):
        header = json.loads(raw_payload.decode("utf-8"))
        return EncodedPayload(
            table_log=int(header["table_log"]),
            original_size=int(header["original_size"]),
            symbols=[int(symbol) for symbol in header["symbols"]],
            frequencies=[int(frequency) for frequency in header["frequencies"]],
            bitstream=unpack_bits(base64.b64decode(header["bitstream"]), int(header["bit_count"])),
        )


class TansCoder:
    def __init__(self, table_log, symbols, frequencies, fast_spread=True):
        self.table_log = table_log
        self.table_size = 1 << table_log
        self.symbols = list(symbols)
        self.frequencies = list(frequencies)
        self.fast_spread = fast_spread

        if len(self.symbols) != len(self.frequencies):
            raise ValueError("symbols and frequencies must have the same length")
        if sum(self.frequencies) != self.table_size:
            raise ValueError("frequencies must sum to L")
        if any(frequency <= 0 for frequency in self.frequencies):
            raise ValueError("all frequencies must be positive")

        self.symbol_to_index = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.symbol_spread = self._build_symbol_spread()
        self.decode_table = self._build_decode_table()
        self.encoding_table, self.starts, self.nb_values = self._build_encoding_table()

    def encode(self, data):
        if not data:
            return []

        symbol_indexes = [self.symbol_to_index[symbol] for symbol in data]
        state = 0
        _, state = self._encode_step(state, symbol_indexes[0])
        original_state = state

        bitstream = []
        for symbol_index in symbol_indexes:
            bits, state = self._encode_step(state, symbol_index)
            bitstream.extend(bits)

        bitstream.extend(int_to_bits(state - self.table_size, self.table_log))
        bitstream.extend(int_to_bits(original_state - self.table_size, self.table_log))
        return bitstream

    def decode(self, bitstream):
        if not bitstream:
            return []

        original_state_bits, remaining_bits = read_bits_from_end(bitstream, self.table_log)
        state_bits, remaining_bits = read_bits_from_end(remaining_bits, self.table_log)
        original_state = bits_to_int(original_state_bits)
        state = bits_to_int(state_bits)

        decoded = []
        while remaining_bits or state != original_state:
            entry = self.decode_table[state]
            bits, remaining_bits = read_bits_from_end(remaining_bits, entry.nb_bits)
            state = entry.new_state + bits_to_int(bits)
            decoded.append(self.symbols[entry.symbol_index])

        decoded.reverse()
        return decoded

    def _build_symbol_spread(self):
        if self.fast_spread:
            return self._build_fast_spread()
        return self._build_slow_spread()

    def _build_fast_spread(self):
        spread = [0] * self.table_size
        position = 0
        step = (self.table_size >> 1) + (self.table_size >> 3) + 3
        while math.gcd(step, self.table_size) != 1:
            step += 2

        for symbol_index, frequency in enumerate(self.frequencies):
            for _ in range(frequency):
                spread[position] = symbol_index
                position = (position + step) % self.table_size

        return spread

    def _build_slow_spread(self):
        weighted_positions = []
        for symbol_index, frequency in enumerate(self.frequencies):
            probability = float(frequency) / float(self.table_size)
            for position_index in range(self.table_size):
                weighted_position = 1.0 / (2.0 * probability) + position_index / probability
                weighted_positions.append((weighted_position, symbol_index))

        weighted_positions.sort()
        return [symbol_index for _, symbol_index in weighted_positions[: self.table_size]]

    def _build_decode_table(self):
        next_values = list(self.frequencies)
        decode_table = []

        for state in range(self.table_size):
            symbol_index = self.symbol_spread[state]
            temporary_state = next_values[symbol_index]
            next_values[symbol_index] += 1

            nb_bits = self.table_log - int(math.floor(math.log(temporary_state, 2)))
            new_state = (temporary_state << nb_bits) - self.table_size
            decode_table.append(DecodeEntry(symbol_index, nb_bits, new_state))

        return decode_table

    def _build_encoding_table(self):
        starts = []
        cumulative_frequency = 0
        for frequency in self.frequencies:
            starts.append(cumulative_frequency - frequency)
            cumulative_frequency += frequency

        next_values = list(self.frequencies)
        encoding_table = [0] * self.table_size
        for table_position, symbol_index in enumerate(self.symbol_spread):
            encoding_table[starts[symbol_index] + next_values[symbol_index]] = table_position + self.table_size
            next_values[symbol_index] += 1

        r_value = self.table_log + 1
        nb_values = []
        for frequency in self.frequencies:
            k_value = self.table_log - int(math.floor(math.log(frequency, 2)))
            nb_values.append((k_value << r_value) - (frequency << k_value))

        return encoding_table, starts, nb_values

    def _encode_step(self, state, symbol_index):
        r_value = self.table_log + 1
        nb_bits = max(0, (state + self.nb_values[symbol_index]) >> r_value)
        bits = int_to_bits(state & ((1 << nb_bits) - 1), nb_bits) if nb_bits > 0 else []
        reduced_state = state >> nb_bits if nb_bits > 0 else state
        next_state = self.encoding_table[self.starts[symbol_index] + reduced_state]
        return bits, next_state


def normalize_frequencies(data, table_log):
    if not data:
        return [], []

    table_size = 1 << table_log
    symbol_counts = collections.Counter(data)
    if len(symbol_counts) > table_size:
        raise ValueError("table is too small for the number of distinct symbols")

    symbols = sorted(symbol_counts)
    frequencies_by_symbol = {}
    fractional_parts = []
    assigned_total = 0

    for symbol in symbols:
        exact_frequency = symbol_counts[symbol] * table_size / float(len(data))
        base_frequency = max(1, int(math.floor(exact_frequency)))
        frequencies_by_symbol[symbol] = base_frequency
        assigned_total += base_frequency
        fractional_parts.append((exact_frequency - math.floor(exact_frequency), symbol))

    difference = table_size - assigned_total
    if difference > 0:
        for _, symbol in sorted(fractional_parts, reverse=True)[:difference]:
            frequencies_by_symbol[symbol] += 1
    elif difference < 0:
        removable_heap = [
            (fractional_part, symbol)
            for fractional_part, symbol in fractional_parts
            if frequencies_by_symbol[symbol] > 1
        ]
        heapq.heapify(removable_heap)

        for _ in range(-difference):
            if not removable_heap:
                raise ValueError("could not normalize frequencies without dropping a symbol")
            _, symbol = heapq.heappop(removable_heap)
            frequencies_by_symbol[symbol] -= 1
            if frequencies_by_symbol[symbol] > 1:
                heapq.heappush(removable_heap, (0.0, symbol))

    return symbols, [frequencies_by_symbol[symbol] for symbol in symbols]


def tans_encode(data, table_log=DEFAULT_TABLE_LOG, fast_spread=True):
    if not (MIN_TABLE_LOG <= table_log <= MAX_TABLE_LOG):
        raise ValueError("table_log must be between %d and %d" % (MIN_TABLE_LOG, MAX_TABLE_LOG))

    if not data:
        return EncodedPayload(table_log, 0, [], [], [])

    byte_symbols = list(bytearray(data))
    symbols, frequencies = normalize_frequencies(byte_symbols, table_log)
    coder = TansCoder(table_log, symbols, frequencies, fast_spread=fast_spread)
    return EncodedPayload(table_log, len(data), symbols, frequencies, coder.encode(byte_symbols))


def tans_decode(payload):
    if payload.original_size == 0:
        return b""

    coder = TansCoder(payload.table_log, payload.symbols, payload.frequencies)
    decoded_symbols = coder.decode(payload.bitstream)
    if len(decoded_symbols) != payload.original_size:
        raise ValueError("decoded size does not match payload header")
    return bytes(bytearray(decoded_symbols))


def int_to_bits(value, bit_count):
    if bit_count == 0:
        return []
    return [int(bit) for bit in bin(value)[2:].zfill(bit_count)]


def bits_to_int(bits):
    if not bits:
        return 0
    return int("".join(str(bit) for bit in bits), 2)


def read_bits_from_end(bitstream, bit_count):
    if bit_count == 0:
        return [], bitstream
    if len(bitstream) < bit_count:
        raise ValueError("compressed bitstream ended unexpectedly")
    return bitstream[-bit_count:], bitstream[:-bit_count]


def pack_bits(bits):
    output = bytearray()
    current_byte = 0
    bits_in_current_byte = 0

    for bit in bits:
        current_byte = (current_byte << 1) | int(bit)
        bits_in_current_byte += 1
        if bits_in_current_byte == 8:
            output.append(current_byte)
            current_byte = 0
            bits_in_current_byte = 0

    if bits_in_current_byte:
        current_byte <<= 8 - bits_in_current_byte
        output.append(current_byte)

    return bytes(output)


def unpack_bits(data, bit_count):
    bits = []
    for byte_value in bytearray(data):
        for bit_index in range(7, -1, -1):
            if len(bits) == bit_count:
                return bits
            bits.append((byte_value >> bit_index) & 1)
    return bits


def compress_file(input_path, output_path, table_log):
    with open(input_path, "rb") as input_file:
        data = input_file.read()

    payload = tans_encode(data, table_log=table_log)

    with open(output_path, "wb") as output_file:
        output_file.write(payload.to_bytes())


def decompress_file(input_path, output_path):
    with open(input_path, "rb") as input_file:
        payload = EncodedPayload.from_bytes(input_file.read())

    with open(output_path, "wb") as output_file:
        output_file.write(tans_decode(payload))


def run_demo():
    sample = (
        b"tANS is an entropy coding method. "
        b"This demo follows the adamrt27/tANS_py table construction. "
        b"aaaaabbbbcccdde"
    )
    payload = tans_encode(sample)
    restored = tans_decode(payload)

    if restored != sample:
        raise AssertionError("tANS round-trip failed")

    print("tANS demo round-trip succeeded")
    print("original bytes:   %d" % len(sample))
    print("payload bits:     %d" % len(payload.bitstream))
    print("container bytes:  %d" % len(payload.to_bytes()))
    print("table log:        %d" % payload.table_log)
    print("symbols:          %d" % len(payload.symbols))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Educational tANS lossless compressor")
    subparsers = parser.add_subparsers(dest="command")

    compress_parser = subparsers.add_parser("compress", help="compress a file")
    compress_parser.add_argument("input_path")
    compress_parser.add_argument("output_path")
    compress_parser.add_argument("--table-log", type=int, default=DEFAULT_TABLE_LOG)

    decompress_parser = subparsers.add_parser("decompress", help="decompress a file")
    decompress_parser.add_argument("input_path")
    decompress_parser.add_argument("output_path")

    subparsers.add_parser("demo", help="run an in-memory round-trip demo")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "compress":
        compress_file(args.input_path, args.output_path, args.table_log)
        return

    if args.command == "decompress":
        decompress_file(args.input_path, args.output_path)
        return

    run_demo()


if __name__ == "__main__":
    main()
