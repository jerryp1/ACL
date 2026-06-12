#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::uint32_t kDefaultTableLog = 8;
constexpr std::uint32_t kFastTableLog = 8;
constexpr std::uint32_t kFastTableSize = 1u << kFastTableLog;
constexpr std::uint32_t kMinTableLog = 7;
constexpr std::uint32_t kMaxTableLog = 15;
constexpr std::uint32_t kF32Bits = 32;
constexpr std::uint32_t kF32Symbols = 2 * kF32Bits + 1;
constexpr std::uint32_t kF32Bias = kF32Bits;
constexpr std::uint8_t kMagic[4] = {'D', 'F', 'T', 0x02};
constexpr std::size_t kContainerHeaderSize = 4 + 8 + 1 + 1 + 8 + 8 + 8 + 4 + 4;
constexpr std::size_t kFrequencyEntrySize = 3;

struct DecodeEntry {
    std::uint8_t symbol;
    std::uint8_t nb_bits;
    std::uint32_t new_state;
};

struct PackedBitStream {
    std::vector<std::uint8_t> bytes;
    std::uint64_t bit_count = 0;
};

struct TansPayload {
    std::uint8_t table_log = 0;
    std::uint64_t original_size = 0;
    std::vector<std::uint8_t> symbols;
    std::vector<std::uint16_t> frequencies;
    std::vector<std::uint8_t> bitstream;
    std::uint64_t bit_count = 0;
};

struct ParsedPayload {
    std::uint64_t element_count = 0;
    TansPayload joint_payload;
    std::vector<std::uint8_t> mantissa_stream;
    std::uint64_t mantissa_bit_count = 0;
};

static std::uint16_t read_u16_le(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(data[1] << 8);
}

static std::uint32_t read_u32_le(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

static std::uint64_t read_u64_le(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (std::uint32_t byte_index = 0; byte_index < 8; ++byte_index) {
        value |= static_cast<std::uint64_t>(data[byte_index]) << (8 * byte_index);
    }
    return value;
}

static void append_u16_le(std::vector<std::uint8_t>& output, std::uint16_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xffu));
    output.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
}

static void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (std::uint32_t byte_index = 0; byte_index < 4; ++byte_index) {
        output.push_back(static_cast<std::uint8_t>((value >> (8 * byte_index)) & 0xffu));
    }
}

static void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (std::uint32_t byte_index = 0; byte_index < 8; ++byte_index) {
        output.push_back(static_cast<std::uint8_t>((value >> (8 * byte_index)) & 0xffu));
    }
}

static void write_u32_le(std::uint8_t* output, std::uint32_t value) {
    output[0] = static_cast<std::uint8_t>(value & 0xffu);
    output[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
    output[2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
    output[3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
}

static std::uint32_t rotl1_u32(std::uint32_t value) {
    return ((value << 1u) & 0xffffffffu) | (value >> 31u);
}

static std::uint32_t rotr1_u32(std::uint32_t value) {
    return (value >> 1u) | ((value & 1u) << 31u);
}

static std::uint32_t bsr32(std::uint32_t value) {
    if (value == 0) {
        throw std::invalid_argument("bsr32 expects a positive value");
    }
    return 31u - static_cast<std::uint32_t>(__builtin_clz(value));
}

static std::uint32_t low_bits_mask(std::uint32_t bit_count) {
    if (bit_count == 32u) {
        return 0xffffffffu;
    }
    return (1u << bit_count) - 1u;
}

static std::uint32_t read_msb_bits_window(
    const std::vector<std::uint8_t>& bytes,
    std::uint64_t absolute_bit,
    std::uint32_t bit_count) {
    if (bit_count == 0) {
        return 0;
    }

    const std::uint32_t bit_offset = static_cast<std::uint32_t>(absolute_bit & 7u);
    const std::uint32_t bytes_to_load = (bit_offset + bit_count + 7u) >> 3u;
    const std::size_t byte_index = static_cast<std::size_t>(absolute_bit >> 3u);

    std::uint64_t window = 0;
    for (std::uint32_t index = 0; index < bytes_to_load; ++index) {
        window = (window << 8u) | bytes[byte_index + index];
    }

    const std::uint32_t loaded_bits = bytes_to_load * 8u;
    const std::uint32_t shift = loaded_bits - bit_offset - bit_count;
    return static_cast<std::uint32_t>((window >> shift) & low_bits_mask(bit_count));
}

class PackedBitWriter {
public:
    void reserve_bits(std::uint64_t bit_count) {
        bytes_.reserve(static_cast<std::size_t>((bit_count + 7u) / 8u));
    }

    void write_bits(std::uint32_t value, std::uint32_t bit_count) {
        if (bit_count == 0) {
            return;
        }

        bit_count_ += bit_count;
        bit_buffer_ = (bit_buffer_ << bit_count) | (static_cast<std::uint64_t>(value) & low_bits_mask(bit_count));
        bits_in_buffer_ += bit_count;

        while (bits_in_buffer_ >= 8u) {
            const std::uint32_t shift = bits_in_buffer_ - 8u;
            bytes_.push_back(static_cast<std::uint8_t>((bit_buffer_ >> shift) & 0xffu));
            bits_in_buffer_ -= 8u;
            bit_buffer_ &= low_bits_mask(bits_in_buffer_);
        }
    }

    void flush() {
        if (bits_in_buffer_ == 0) {
            return;
        }

        bytes_.push_back(static_cast<std::uint8_t>((bit_buffer_ << (8u - bits_in_buffer_)) & 0xffu));
        bit_buffer_ = 0;
        bits_in_buffer_ = 0;
    }

    const std::vector<std::uint8_t>& bytes() {
        flush();
        return bytes_;
    }

    std::uint64_t bit_count() const {
        return bit_count_;
    }

private:
    std::vector<std::uint8_t> bytes_;
    std::uint64_t bit_buffer_ = 0;
    std::uint32_t bits_in_buffer_ = 0;
    std::uint64_t bit_count_ = 0;
};

class PackedBitReader {
public:
    PackedBitReader(const std::vector<std::uint8_t>& bytes, std::uint64_t bit_count)
        : bytes_(bytes), bit_count_(bit_count) {}

    std::uint32_t read_bits(std::uint32_t bit_count) {
        if (position_ + bit_count > bit_count_) {
            throw std::invalid_argument("mantissa bitstream ended unexpectedly");
        }

        const std::uint32_t value = read_msb_bits_window(bytes_, position_, bit_count);
        position_ += bit_count;
        return value;
    }

    void assert_consumed() const {
        if (position_ != bit_count_) {
            throw std::invalid_argument("mantissa bitstream has unused bits");
        }
    }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::uint64_t bit_count_ = 0;
    std::uint64_t position_ = 0;
};

class PackedTailBitReader {
public:
    PackedTailBitReader(const std::vector<std::uint8_t>& bytes, std::uint64_t bit_count)
        : bytes_(bytes), read_position_(bit_count) {}

    std::uint32_t read_bits_from_tail(std::uint32_t bit_count) {
        if (read_position_ < bit_count) {
            throw std::invalid_argument("compressed bitstream ended unexpectedly");
        }

        read_position_ -= bit_count;
        return read_msb_bits_window(bytes_, read_position_, bit_count);
    }

    void assert_consumed() const {
        if (read_position_ != 0) {
            throw std::invalid_argument("decoded tANS stream has unused bits");
        }
    }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::uint64_t read_position_ = 0;
};

class TansCoder {
public:
    TansCoder(std::uint8_t table_log, std::vector<std::uint8_t> symbols, std::vector<std::uint16_t> frequencies)
        : table_log_(table_log),
          table_size_(1u << table_log),
          symbols_(std::move(symbols)),
          frequencies_(std::move(frequencies)) {
        if (symbols_.size() != frequencies_.size()) {
            throw std::invalid_argument("symbols and frequencies must have the same length");
        }
        if (symbols_.empty()) {
            throw std::invalid_argument("non-empty alphabet is required");
        }

        std::uint32_t total_frequency = 0;
        for (std::uint16_t frequency : frequencies_) {
            if (frequency == 0) {
                throw std::invalid_argument("all frequencies must be positive");
            }
            total_frequency += frequency;
        }
        if (total_frequency != table_size_) {
            throw std::invalid_argument("frequencies must sum to table size");
        }

        for (std::size_t index = 0; index < symbols_.size(); ++index) {
            symbol_to_index_[symbols_[index]] = static_cast<std::uint16_t>(index);
            symbol_is_present_[symbols_[index]] = true;
        }

        build_fast_spread();
        build_decode_table();
        build_encoding_table();
    }

    PackedBitStream encode(const std::vector<std::uint8_t>& data) const {
        if (data.empty()) {
            return {};
        }

        std::uint32_t state = table_size_;
        const std::uint32_t original_state = state;
        PackedBitWriter bitstream_writer;
        bitstream_writer.reserve_bits(data.size() * 8u + 2u * table_log_);

        for (std::uint8_t symbol : data) {
            if (symbol >= kF32Symbols || !symbol_is_present_[symbol]) {
                throw std::invalid_argument("symbol is not in alphabet");
            }
            encode_step(state, symbol_to_index_[symbol], bitstream_writer);
        }

        bitstream_writer.write_bits(state - table_size_, table_log_);
        bitstream_writer.write_bits(original_state - table_size_, table_log_);

        PackedBitStream packed_stream;
        packed_stream.bytes = bitstream_writer.bytes();
        packed_stream.bit_count = bitstream_writer.bit_count();
        return packed_stream;
    }

    template <typename SymbolConsumer>
    void decode_reverse(
        const std::vector<std::uint8_t>& bitstream,
        std::uint64_t bit_count,
        std::uint64_t expected_size,
        SymbolConsumer&& consume_symbol) const {
        if (expected_size == 0) {
            return;
        }
        if (bit_count < 2u * table_log_) {
            throw std::invalid_argument("compressed bitstream is too small");
        }

        PackedTailBitReader bitstream_reader(bitstream, bit_count);
        const std::uint32_t original_state = bitstream_reader.read_bits_from_tail(table_log_);
        std::uint32_t state = bitstream_reader.read_bits_from_tail(table_log_);

        for (std::uint64_t reverse_index = 0; reverse_index < expected_size; ++reverse_index) {
            if (state >= decode_table_.size()) {
                throw std::invalid_argument("invalid tANS decoder state");
            }
            const DecodeEntry& entry = decode_table_[state];
            const std::uint32_t bits = bitstream_reader.read_bits_from_tail(entry.nb_bits);
            state = entry.new_state + bits;

            const std::uint64_t output_index = expected_size - 1u - reverse_index;
            consume_symbol(output_index, entry.symbol);
        }

        if (state != original_state) {
            throw std::invalid_argument("decoded tANS stream ended in an invalid state");
        }
        bitstream_reader.assert_consumed();
    }

    std::vector<std::uint8_t> decode(
        const std::vector<std::uint8_t>& bitstream,
        std::uint64_t bit_count,
        std::uint64_t expected_size) const {
        std::vector<std::uint8_t> decoded(static_cast<std::size_t>(expected_size));
        decode_reverse(bitstream, bit_count, expected_size, [&](std::uint64_t output_index, std::uint8_t symbol) {
            decoded[static_cast<std::size_t>(output_index)] = symbol;
        });
        return decoded;
    }

private:
    void build_fast_spread() {
        symbol_spread_.assign(table_size_, 0);

        std::uint32_t position = 0;
        std::uint32_t step = (table_size_ >> 1u) + (table_size_ >> 3u) + 3u;
        while (std::gcd(step, table_size_) != 1u) {
            step += 2u;
        }

        for (std::uint16_t symbol_index = 0; symbol_index < frequencies_.size(); ++symbol_index) {
            for (std::uint16_t count = 0; count < frequencies_[symbol_index]; ++count) {
                symbol_spread_[position] = symbol_index;
                position = (position + step) % table_size_;
            }
        }
    }

    void build_decode_table() {
        std::vector<std::uint32_t> next_values(frequencies_.begin(), frequencies_.end());
        decode_table_.clear();
        decode_table_.reserve(table_size_);

        for (std::uint32_t state = 0; state < table_size_; ++state) {
            const std::uint16_t symbol_index = symbol_spread_[state];
            const std::uint32_t temporary_state = next_values[symbol_index]++;
            const std::uint8_t nb_bits = static_cast<std::uint8_t>(table_log_ - bsr32(temporary_state));
            const std::uint32_t new_state = (temporary_state << nb_bits) - table_size_;
            decode_table_.push_back(DecodeEntry{symbols_[symbol_index], nb_bits, new_state});
        }
    }

    void build_encoding_table() {
        starts_.assign(frequencies_.size(), 0);
        std::int32_t cumulative_frequency = 0;
        for (std::size_t index = 0; index < frequencies_.size(); ++index) {
            starts_[index] = cumulative_frequency - static_cast<std::int32_t>(frequencies_[index]);
            cumulative_frequency += frequencies_[index];
        }

        std::vector<std::uint32_t> next_values(frequencies_.begin(), frequencies_.end());
        encoding_table_.assign(table_size_, 0);
        for (std::uint32_t table_position = 0; table_position < table_size_; ++table_position) {
            const std::uint16_t symbol_index = symbol_spread_[table_position];
            const std::int32_t encoding_index = starts_[symbol_index] + static_cast<std::int32_t>(next_values[symbol_index]);
            encoding_table_[encoding_index] = table_position + table_size_;
            ++next_values[symbol_index];
        }

        const std::uint32_t r_value = table_log_ + 1u;
        nb_values_.assign(frequencies_.size(), 0);
        for (std::size_t index = 0; index < frequencies_.size(); ++index) {
            const std::uint32_t frequency = frequencies_[index];
            const std::uint32_t k_value = table_log_ - bsr32(frequency);
            nb_values_[index] = (static_cast<std::int32_t>(k_value) << r_value) -
                                static_cast<std::int32_t>(frequency << k_value);
        }
    }

    void encode_step(std::uint32_t& state, std::uint16_t symbol_index, PackedBitWriter& bitstream_writer) const {
        const std::uint32_t r_value = table_log_ + 1u;
        const std::uint32_t nb_bits = static_cast<std::uint32_t>(
            (static_cast<std::int64_t>(state) + nb_values_[symbol_index]) >> r_value);
        bitstream_writer.write_bits(state & ((1u << nb_bits) - 1u), nb_bits);
        const std::uint32_t reduced_state = state >> nb_bits;
        state = encoding_table_[starts_[symbol_index] + static_cast<std::int32_t>(reduced_state)];
    }

    std::uint8_t table_log_;
    std::uint32_t table_size_;
    std::vector<std::uint8_t> symbols_;
    std::vector<std::uint16_t> frequencies_;
    std::array<std::uint16_t, kF32Symbols> symbol_to_index_{};
    std::array<bool, kF32Symbols> symbol_is_present_{};
    std::vector<std::uint16_t> symbol_spread_;
    std::vector<DecodeEntry> decode_table_;
    std::vector<std::uint32_t> encoding_table_;
    std::vector<std::int32_t> starts_;
    std::vector<std::int32_t> nb_values_;
};

class FastTansCoder8 {
public:
    FastTansCoder8(const std::vector<std::uint8_t>& symbols, const std::vector<std::uint16_t>& frequencies) {
        if (symbols.size() != frequencies.size()) {
            throw std::invalid_argument("symbols and frequencies must have the same length");
        }
        if (symbols.empty()) {
            throw std::invalid_argument("non-empty alphabet is required");
        }
        if (symbols.size() > kF32Symbols) {
            throw std::invalid_argument("too many symbols for fast tANS coder");
        }

        symbol_count_ = symbols.size();
        std::uint32_t total_frequency = 0;
        for (std::size_t index = 0; index < symbol_count_; ++index) {
            if (frequencies[index] == 0) {
                throw std::invalid_argument("all frequencies must be positive");
            }
            const std::uint8_t symbol = symbols[index];
            symbol_by_index_[index] = symbol;
            frequency_by_index_[index] = frequencies[index];
            symbol_to_index_[symbol] = static_cast<std::uint16_t>(index);
            symbol_is_present_[symbol] = true;
            total_frequency += frequencies[index];
        }
        if (total_frequency != kFastTableSize) {
            throw std::invalid_argument("frequencies must sum to table size");
        }

        build_fast_spread();
        build_decode_table();
        build_encoding_table();
    }

    PackedBitStream encode(const std::vector<std::uint8_t>& data) const {
        if (data.empty()) {
            return {};
        }

        std::uint32_t state = kFastTableSize;
        const std::uint32_t original_state = state;
        PackedBitWriter bitstream_writer;
        bitstream_writer.reserve_bits(data.size() * 8u + 2u * kFastTableLog);

        for (std::uint8_t symbol : data) {
            if (symbol >= kF32Symbols || !symbol_is_present_[symbol]) {
                throw std::invalid_argument("symbol is not in alphabet");
            }
            encode_step(state, symbol_to_index_[symbol], bitstream_writer);
        }

        bitstream_writer.write_bits(state - kFastTableSize, kFastTableLog);
        bitstream_writer.write_bits(original_state - kFastTableSize, kFastTableLog);

        PackedBitStream packed_stream;
        packed_stream.bytes = bitstream_writer.bytes();
        packed_stream.bit_count = bitstream_writer.bit_count();
        return packed_stream;
    }

    template <typename SymbolConsumer>
    void decode_reverse(
        const std::vector<std::uint8_t>& bitstream,
        std::uint64_t bit_count,
        std::uint64_t expected_size,
        SymbolConsumer&& consume_symbol) const {
        if (expected_size == 0) {
            return;
        }
        if (bit_count < 2u * kFastTableLog) {
            throw std::invalid_argument("compressed bitstream is too small");
        }

        PackedTailBitReader bitstream_reader(bitstream, bit_count);
        const std::uint32_t original_state = bitstream_reader.read_bits_from_tail(kFastTableLog);
        std::uint32_t state = bitstream_reader.read_bits_from_tail(kFastTableLog);

        for (std::uint64_t reverse_index = 0; reverse_index < expected_size; ++reverse_index) {
            if (state >= kFastTableSize) {
                throw std::invalid_argument("invalid tANS decoder state");
            }
            const DecodeEntry& entry = decode_table_[state];
            const std::uint32_t bits = bitstream_reader.read_bits_from_tail(entry.nb_bits);
            state = entry.new_state + bits;

            const std::uint64_t output_index = expected_size - 1u - reverse_index;
            consume_symbol(output_index, entry.symbol);
        }

        if (state != original_state) {
            throw std::invalid_argument("decoded tANS stream ended in an invalid state");
        }
        bitstream_reader.assert_consumed();
    }

private:
    void build_fast_spread() {
        std::uint32_t position = 0;
        std::uint32_t step = (kFastTableSize >> 1u) + (kFastTableSize >> 3u) + 3u;
        while (std::gcd(step, kFastTableSize) != 1u) {
            step += 2u;
        }

        for (std::uint16_t symbol_index = 0; symbol_index < symbol_count_; ++symbol_index) {
            for (std::uint16_t count = 0; count < frequency_by_index_[symbol_index]; ++count) {
                symbol_spread_[position] = symbol_index;
                position = (position + step) & (kFastTableSize - 1u);
            }
        }
    }

    void build_decode_table() {
        std::array<std::uint32_t, kF32Symbols> next_values{};
        for (std::size_t index = 0; index < symbol_count_; ++index) {
            next_values[index] = frequency_by_index_[index];
        }

        for (std::uint32_t state = 0; state < kFastTableSize; ++state) {
            const std::uint16_t symbol_index = symbol_spread_[state];
            const std::uint32_t temporary_state = next_values[symbol_index]++;
            const std::uint8_t nb_bits = static_cast<std::uint8_t>(kFastTableLog - bsr32(temporary_state));
            const std::uint32_t new_state = (temporary_state << nb_bits) - kFastTableSize;
            decode_table_[state] = DecodeEntry{symbol_by_index_[symbol_index], nb_bits, new_state};
        }
    }

    void build_encoding_table() {
        std::int32_t cumulative_frequency = 0;
        for (std::size_t index = 0; index < symbol_count_; ++index) {
            starts_[index] = cumulative_frequency - static_cast<std::int32_t>(frequency_by_index_[index]);
            cumulative_frequency += frequency_by_index_[index];
        }

        std::array<std::uint32_t, kF32Symbols> next_values{};
        for (std::size_t index = 0; index < symbol_count_; ++index) {
            next_values[index] = frequency_by_index_[index];
        }

        for (std::uint32_t table_position = 0; table_position < kFastTableSize; ++table_position) {
            const std::uint16_t symbol_index = symbol_spread_[table_position];
            const std::int32_t encoding_index = starts_[symbol_index] + static_cast<std::int32_t>(next_values[symbol_index]);
            encoding_table_[encoding_index] = table_position + kFastTableSize;
            ++next_values[symbol_index];
        }

        constexpr std::uint32_t r_value = kFastTableLog + 1u;
        for (std::size_t index = 0; index < symbol_count_; ++index) {
            const std::uint32_t frequency = frequency_by_index_[index];
            const std::uint32_t k_value = kFastTableLog - bsr32(frequency);
            nb_values_[index] = (static_cast<std::int32_t>(k_value) << r_value) -
                                static_cast<std::int32_t>(frequency << k_value);
        }
    }

    void encode_step(std::uint32_t& state, std::uint16_t symbol_index, PackedBitWriter& bitstream_writer) const {
        constexpr std::uint32_t r_value = kFastTableLog + 1u;
        const std::uint32_t nb_bits = static_cast<std::uint32_t>(
            (static_cast<std::int64_t>(state) + nb_values_[symbol_index]) >> r_value);
        bitstream_writer.write_bits(state & low_bits_mask(nb_bits), nb_bits);
        const std::uint32_t reduced_state = state >> nb_bits;
        state = encoding_table_[starts_[symbol_index] + static_cast<std::int32_t>(reduced_state)];
    }

    std::size_t symbol_count_ = 0;
    std::array<std::uint8_t, kF32Symbols> symbol_by_index_{};
    std::array<std::uint16_t, kF32Symbols> frequency_by_index_{};
    std::array<std::uint16_t, kF32Symbols> symbol_to_index_{};
    std::array<bool, kF32Symbols> symbol_is_present_{};
    std::array<std::uint16_t, kFastTableSize> symbol_spread_{};
    std::array<DecodeEntry, kFastTableSize> decode_table_{};
    std::array<std::uint32_t, kFastTableSize> encoding_table_{};
    std::array<std::int32_t, kF32Symbols> starts_{};
    std::array<std::int32_t, kF32Symbols> nb_values_{};
};

static std::pair<std::uint8_t, std::pair<std::uint32_t, std::uint32_t>> encode_pc_symbol_f32(
    std::uint32_t reference_rotl,
    std::uint32_t target_rotl) {
    if (reference_rotl < target_rotl) {
        const std::uint32_t delta = target_rotl - reference_rotl;
        const std::uint32_t level = bsr32(delta);
        return {static_cast<std::uint8_t>(kF32Bias + 1u + level), {delta - (1u << level), level}};
    }

    if (reference_rotl > target_rotl) {
        const std::uint32_t delta = reference_rotl - target_rotl;
        const std::uint32_t level = bsr32(delta);
        return {static_cast<std::uint8_t>(kF32Bias - 1u - level), {delta - (1u << level), level}};
    }

    return {static_cast<std::uint8_t>(kF32Bias), {0, 0}};
}

static std::uint32_t decode_pc_symbol_f32_from_tail(
    std::uint32_t reference_rotl,
    std::uint8_t symbol,
    PackedTailBitReader& mantissa_reader) {
    if (symbol == kF32Bias) {
        return reference_rotl;
    }

    if (symbol > kF32Bias) {
        const std::uint32_t level = symbol - kF32Bias - 1u;
        const std::uint32_t delta = (1u << level) + mantissa_reader.read_bits_from_tail(level);
        return reference_rotl + delta;
    }

    const std::uint32_t level = kF32Bias - 1u - symbol;
    const std::uint32_t delta = (1u << level) + mantissa_reader.read_bits_from_tail(level);
    return reference_rotl - delta;
}

static std::pair<std::vector<std::uint8_t>, std::vector<std::uint16_t>> normalize_frequencies(
    const std::array<std::uint32_t, kF32Symbols>& counts,
    std::uint64_t data_size,
    std::uint8_t table_log) {
    const std::uint32_t table_size = 1u << table_log;

    std::vector<std::uint8_t> symbols;
    for (std::uint32_t symbol = 0; symbol < kF32Symbols; ++symbol) {
        if (counts[symbol] > 0) {
            symbols.push_back(static_cast<std::uint8_t>(symbol));
        }
    }
    if (symbols.size() > table_size) {
        throw std::invalid_argument("table is too small for alphabet");
    }

    std::array<std::uint16_t, kF32Symbols> frequencies{};
    struct FractionalPart {
        double fraction;
        std::uint8_t symbol;
    };
    std::vector<FractionalPart> fractional_parts;
    std::int32_t assigned_total = 0;

    for (std::uint8_t symbol : symbols) {
        const double exact_frequency = static_cast<double>(counts[symbol]) *
                                       static_cast<double>(table_size) /
                                       static_cast<double>(data_size);
        const std::uint16_t base_frequency = static_cast<std::uint16_t>(
            std::max(1, static_cast<int>(std::floor(exact_frequency))));
        frequencies[symbol] = base_frequency;
        assigned_total += base_frequency;
        fractional_parts.push_back(FractionalPart{exact_frequency - std::floor(exact_frequency), symbol});
    }

    const std::int32_t difference = static_cast<std::int32_t>(table_size) - assigned_total;
    if (difference > 0) {
        std::sort(fractional_parts.begin(), fractional_parts.end(), [](const FractionalPart& left, const FractionalPart& right) {
            if (left.fraction != right.fraction) {
                return left.fraction > right.fraction;
            }
            return left.symbol > right.symbol;
        });
        for (std::int32_t index = 0; index < difference; ++index) {
            ++frequencies[fractional_parts[index].symbol];
        }
    } else if (difference < 0) {
        using HeapItem = std::pair<double, std::uint8_t>;
        std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> removable_heap;
        for (const FractionalPart& item : fractional_parts) {
            if (frequencies[item.symbol] > 1) {
                removable_heap.push({item.fraction, item.symbol});
            }
        }

        for (std::int32_t index = 0; index < -difference; ++index) {
            if (removable_heap.empty()) {
                throw std::invalid_argument("could not normalize frequencies without dropping a symbol");
            }
            const auto item = removable_heap.top();
            removable_heap.pop();
            --frequencies[item.second];
            if (frequencies[item.second] > 1) {
                removable_heap.push({0.0, item.second});
            }
        }
    }

    std::vector<std::uint16_t> output_frequencies;
    output_frequencies.reserve(symbols.size());
    for (std::uint8_t symbol : symbols) {
        output_frequencies.push_back(frequencies[symbol]);
    }
    return {symbols, output_frequencies};
}

static TansPayload tans_encode_symbols(
    const std::vector<std::uint8_t>& symbols,
    const std::array<std::uint32_t, kF32Symbols>& symbol_counts,
    std::uint8_t table_log) {
    TansPayload payload;
    payload.table_log = table_log;
    payload.original_size = symbols.size();

    if (symbols.empty()) {
        return payload;
    }

    auto normalized = normalize_frequencies(symbol_counts, symbols.size(), table_log);
    payload.symbols = std::move(normalized.first);
    payload.frequencies = std::move(normalized.second);

    PackedBitStream packed_stream;
    if (payload.table_log == kFastTableLog) {
        const FastTansCoder8 coder(payload.symbols, payload.frequencies);
        packed_stream = coder.encode(symbols);
    } else {
        const TansCoder coder(payload.table_log, payload.symbols, payload.frequencies);
        packed_stream = coder.encode(symbols);
    }

    payload.bitstream = std::move(packed_stream.bytes);
    payload.bit_count = packed_stream.bit_count;
    return payload;
}

static std::vector<std::uint8_t> pack_payload(
    std::uint64_t element_count,
    const TansPayload& joint_payload,
    const std::vector<std::uint8_t>& mantissa_stream,
    std::uint64_t mantissa_bit_count) {
    if (joint_payload.symbols.size() > kF32Symbols) {
        throw std::invalid_argument("too many joint symbols for f32 payload");
    }

    const std::vector<std::uint8_t>& joint_stream = joint_payload.bitstream;

    std::vector<std::uint8_t> output;
    output.reserve(kContainerHeaderSize + joint_payload.symbols.size() * kFrequencyEntrySize +
                   joint_stream.size() + mantissa_stream.size());

    output.insert(output.end(), std::begin(kMagic), std::end(kMagic));
    append_u64_le(output, element_count);
    output.push_back(joint_payload.table_log);
    output.push_back(static_cast<std::uint8_t>(joint_payload.symbols.size()));
    append_u64_le(output, joint_payload.original_size);
    append_u64_le(output, joint_payload.bit_count);
    append_u64_le(output, mantissa_bit_count);
    append_u32_le(output, static_cast<std::uint32_t>(joint_stream.size()));
    append_u32_le(output, static_cast<std::uint32_t>(mantissa_stream.size()));

    for (std::size_t index = 0; index < joint_payload.symbols.size(); ++index) {
        output.push_back(joint_payload.symbols[index]);
        append_u16_le(output, joint_payload.frequencies[index]);
    }

    output.insert(output.end(), joint_stream.begin(), joint_stream.end());
    output.insert(output.end(), mantissa_stream.begin(), mantissa_stream.end());
    return output;
}

static ParsedPayload unpack_payload(std::string_view encoded_bytes) {
    if (encoded_bytes.size() < kContainerHeaderSize) {
        throw std::invalid_argument("encoded payload is too small");
    }

    const auto* data = reinterpret_cast<const std::uint8_t*>(encoded_bytes.data());
    if (!std::equal(std::begin(kMagic), std::end(kMagic), data)) {
        throw std::invalid_argument("invalid drotl1fmd-tANS magic");
    }

    ParsedPayload parsed;
    std::size_t offset = 4;
    parsed.element_count = read_u64_le(data + offset);
    offset += 8;
    parsed.joint_payload.table_log = data[offset++];
    const std::uint8_t symbol_count = data[offset++];
    parsed.joint_payload.original_size = read_u64_le(data + offset);
    offset += 8;
    const std::uint64_t joint_bit_count = read_u64_le(data + offset);
    offset += 8;
    const std::uint64_t mantissa_bit_count = read_u64_le(data + offset);
    offset += 8;
    const std::uint32_t joint_stream_size = read_u32_le(data + offset);
    offset += 4;
    const std::uint32_t mantissa_stream_size = read_u32_le(data + offset);
    offset += 4;

    if (parsed.joint_payload.table_log < kMinTableLog || parsed.joint_payload.table_log > kMaxTableLog) {
        throw std::invalid_argument("invalid table_log in encoded payload");
    }
    if (symbol_count > kF32Symbols) {
        throw std::invalid_argument("too many joint symbols in encoded payload");
    }
    if (parsed.joint_payload.original_size != parsed.element_count) {
        throw std::invalid_argument("joint symbol count does not match element count");
    }
    if (joint_bit_count > static_cast<std::uint64_t>(joint_stream_size) * 8u) {
        throw std::invalid_argument("joint bit count exceeds joint stream byte length");
    }
    if (mantissa_bit_count > static_cast<std::uint64_t>(mantissa_stream_size) * 8u) {
        throw std::invalid_argument("mantissa bit count exceeds mantissa stream byte length");
    }
    if (parsed.element_count == 0 && symbol_count != 0) {
        throw std::invalid_argument("empty payload must not contain joint symbols");
    }
    if (parsed.element_count == 0 && (joint_bit_count != 0 || joint_stream_size != 0 ||
                                      mantissa_bit_count != 0 || mantissa_stream_size != 0)) {
        throw std::invalid_argument("empty payload must not contain bitstreams");
    }
    if (parsed.element_count != 0 && symbol_count == 0) {
        throw std::invalid_argument("non-empty payload must contain joint symbols");
    }

    const std::size_t payload_size = kContainerHeaderSize +
                                     static_cast<std::size_t>(symbol_count) * kFrequencyEntrySize +
                                     joint_stream_size +
                                     mantissa_stream_size;
    if (encoded_bytes.size() != payload_size) {
        throw std::invalid_argument("encoded payload size does not match header");
    }

    std::array<bool, kF32Symbols> seen_symbols{};
    std::uint32_t total_frequency = 0;
    parsed.joint_payload.symbols.reserve(symbol_count);
    parsed.joint_payload.frequencies.reserve(symbol_count);
    for (std::uint8_t index = 0; index < symbol_count; ++index) {
        const std::uint8_t symbol = data[offset++];
        const std::uint16_t frequency = read_u16_le(data + offset);
        offset += 2;

        if (symbol >= kF32Symbols) {
            throw std::invalid_argument("joint symbol is out of range");
        }
        if (seen_symbols[symbol]) {
            throw std::invalid_argument("duplicate joint symbol in encoded payload");
        }
        if (frequency == 0) {
            throw std::invalid_argument("joint symbol frequency must be positive");
        }

        seen_symbols[symbol] = true;
        total_frequency += frequency;
        parsed.joint_payload.symbols.push_back(symbol);
        parsed.joint_payload.frequencies.push_back(frequency);
    }
    if (parsed.element_count != 0 && total_frequency != (1u << parsed.joint_payload.table_log)) {
        throw std::invalid_argument("joint frequencies do not sum to table size");
    }

    parsed.joint_payload.bitstream.assign(data + offset, data + offset + joint_stream_size);
    parsed.joint_payload.bit_count = joint_bit_count;
    offset += joint_stream_size;
    parsed.mantissa_stream.assign(data + offset, data + offset + mantissa_stream_size);
    parsed.mantissa_bit_count = mantissa_bit_count;
    return parsed;
}

static py::bytes encode_drotl1fmd_tans_f32_cpp(py::bytes reference_bytes, py::bytes target_bytes, std::uint32_t table_log) {
    std::string_view reference_view(reference_bytes);
    std::string_view target_view(target_bytes);

    if (reference_view.size() != target_view.size()) {
        throw std::invalid_argument("reference and target must have the same byte length");
    }
    if (reference_view.size() % 4 != 0) {
        throw std::invalid_argument("float32 byte length must be a multiple of 4");
    }
    if (table_log < kMinTableLog || table_log > kMaxTableLog) {
        throw std::invalid_argument("table_log must be between 7 and 15");
    }

    std::vector<std::uint8_t> packed;
    {
        py::gil_scoped_release release;
        const auto* reference_data = reinterpret_cast<const std::uint8_t*>(reference_view.data());
        const auto* target_data = reinterpret_cast<const std::uint8_t*>(target_view.data());
        const std::uint64_t element_count = reference_view.size() / 4;

        std::vector<std::uint8_t> joint_symbols;
        joint_symbols.reserve(static_cast<std::size_t>(element_count));
        std::array<std::uint32_t, kF32Symbols> joint_symbol_counts{};
        PackedBitWriter mantissa_writer;
        mantissa_writer.reserve_bits(element_count * (kF32Bits - 1u));

        for (std::uint64_t index = 0; index < element_count; ++index) {
            const std::uint32_t reference_value = read_u32_le(reference_data + index * 4);
            const std::uint32_t target_value = read_u32_le(target_data + index * 4);
            const auto encoded_symbol = encode_pc_symbol_f32(rotl1_u32(reference_value), rotl1_u32(target_value));
            joint_symbols.push_back(encoded_symbol.first);
            ++joint_symbol_counts[encoded_symbol.first];
            mantissa_writer.write_bits(encoded_symbol.second.first, encoded_symbol.second.second);
        }

        const TansPayload joint_payload = tans_encode_symbols(
            joint_symbols,
            joint_symbol_counts,
            static_cast<std::uint8_t>(table_log));
        packed = pack_payload(
            element_count,
            joint_payload,
            mantissa_writer.bytes(),
            mantissa_writer.bit_count());
    }
    return py::bytes(reinterpret_cast<const char*>(packed.data()), packed.size());
}

static py::bytes decode_drotl1fmd_tans_f32_cpp(py::bytes encoded_bytes, py::bytes reference_bytes) {
    std::string_view encoded_view(encoded_bytes);
    std::string_view reference_view(reference_bytes);

    std::vector<std::uint8_t> output;
    {
        py::gil_scoped_release release;
        const ParsedPayload parsed = unpack_payload(encoded_view);
        if (reference_view.size() != parsed.element_count * 4) {
            throw std::invalid_argument("reference byte length does not match encoded element count");
        }

        const auto* reference_data = reinterpret_cast<const std::uint8_t*>(reference_view.data());
        output.resize(reference_view.size());

        if (parsed.element_count != 0) {
            const TansCoder coder(
                parsed.joint_payload.table_log,
                parsed.joint_payload.symbols,
                parsed.joint_payload.frequencies);
            PackedTailBitReader mantissa_reader(parsed.mantissa_stream, parsed.mantissa_bit_count);

            coder.decode_reverse(
                parsed.joint_payload.bitstream,
                parsed.joint_payload.bit_count,
                parsed.joint_payload.original_size,
                [&](std::uint64_t element_index, std::uint8_t symbol) {
                    const std::uint32_t reference_value = read_u32_le(reference_data + element_index * 4);
                    const std::uint32_t target_rotl = decode_pc_symbol_f32_from_tail(
                        rotl1_u32(reference_value),
                        symbol,
                        mantissa_reader);
                    const std::uint32_t target_value = rotr1_u32(target_rotl);
                    write_u32_le(output.data() + static_cast<std::size_t>(element_index * 4), target_value);
                });

            mantissa_reader.assert_consumed();
        }
    }
    return py::bytes(reinterpret_cast<const char*>(output.data()), output.size());
}

// ============================================================
// pcmap transforms (IEEE 754 f32 ↔ monotone uint32)
// ============================================================

static std::uint32_t pcmap_u32(std::uint32_t bits) {
    const std::uint32_t sign = bits >> 31u;
    return sign == 0 ? bits + 0x80000000u : ~bits;
}

static std::uint32_t pcmap_inverse_u32(std::uint32_t r) {
    const std::uint32_t mask = (-(r >> 31u)) >> 1u;
    return ~(r ^ mask);
}

// ============================================================
// pcdelta end-to-end encode/decode
// ============================================================

static py::bytes encode_pcdelta_tans_f32_cpp(py::bytes reference_bytes, py::bytes target_bytes, std::uint32_t table_log) {
    std::string_view reference_view(reference_bytes);
    std::string_view target_view(target_bytes);
    if (reference_view.size() != target_view.size()) throw std::invalid_argument("size mismatch");
    if (reference_view.size() % 4 != 0) throw std::invalid_argument("not multiple of 4");
    if (table_log < kMinTableLog || table_log > kMaxTableLog) throw std::invalid_argument("table_log out of range");

    std::vector<std::uint8_t> packed;
    {
        py::gil_scoped_release release;
        const auto* ref_data = reinterpret_cast<const std::uint8_t*>(reference_view.data());
        const auto* tgt_data = reinterpret_cast<const std::uint8_t*>(target_view.data());
        const std::uint64_t n = reference_view.size() / 4;

        std::vector<std::uint8_t> joint_symbols;
        joint_symbols.reserve(static_cast<std::size_t>(n));
        std::array<std::uint32_t, kF32Symbols> joint_symbol_counts{};
        PackedBitWriter mantissa_writer;
        mantissa_writer.reserve_bits(n * (kF32Bits - 1u));

        for (std::uint64_t i = 0; i < n; ++i) {
            const std::uint32_t ref_pc = pcmap_u32(read_u32_le(ref_data + i * 4));
            const std::uint32_t tgt_pc = pcmap_u32(read_u32_le(tgt_data + i * 4));
            const auto enc = encode_pc_symbol_f32(ref_pc, tgt_pc);
            joint_symbols.push_back(enc.first);
            ++joint_symbol_counts[enc.first];
            mantissa_writer.write_bits(enc.second.first, enc.second.second);
        }

        const TansPayload joint_payload = tans_encode_symbols(joint_symbols, joint_symbol_counts, static_cast<std::uint8_t>(table_log));
        packed = pack_payload(n, joint_payload, mantissa_writer.bytes(), mantissa_writer.bit_count());
    }
    return py::bytes(reinterpret_cast<const char*>(packed.data()), packed.size());
}

static py::bytes decode_pcdelta_tans_f32_cpp(py::bytes encoded_bytes, py::bytes reference_bytes) {
    std::string_view encoded_view(encoded_bytes);
    std::string_view reference_view(reference_bytes);

    std::vector<std::uint8_t> output;
    {
        py::gil_scoped_release release;
        const ParsedPayload parsed = unpack_payload(encoded_view);
        if (reference_view.size() != parsed.element_count * 4) throw std::invalid_argument("reference size mismatch");

        const auto* ref_data = reinterpret_cast<const std::uint8_t*>(reference_view.data());
        output.resize(reference_view.size());

        if (parsed.element_count != 0) {
            const TansCoder coder(parsed.joint_payload.table_log, parsed.joint_payload.symbols, parsed.joint_payload.frequencies);
            PackedTailBitReader mantissa_reader(parsed.mantissa_stream, parsed.mantissa_bit_count);

            coder.decode_reverse(
                parsed.joint_payload.bitstream, parsed.joint_payload.bit_count, parsed.joint_payload.original_size,
                [&](std::uint64_t idx, std::uint8_t symbol) {
                    const std::uint32_t ref_pc = pcmap_u32(read_u32_le(ref_data + idx * 4));
                    const std::uint32_t tgt_pc = decode_pc_symbol_f32_from_tail(ref_pc, symbol, mantissa_reader);
                    write_u32_le(output.data() + idx * 4, pcmap_inverse_u32(tgt_pc));
                });
            mantissa_reader.assert_consumed();
        }
    }
    return py::bytes(reinterpret_cast<const char*>(output.data()), output.size());
}

// ============================================================
// Byte-level tANS coder (supports full 256-symbol alphabet)
// ============================================================

static constexpr std::uint32_t kByteMaxSymbols = 256;
static constexpr std::uint8_t kByteTansMagic[4] = {'B', 'T', 'A', 'N'};

class ByteTansCoder {
public:
    ByteTansCoder(std::uint8_t table_log, std::vector<std::uint8_t> symbols, std::vector<std::uint16_t> frequencies)
        : table_log_(table_log), table_size_(1u << table_log),
          symbols_(std::move(symbols)), frequencies_(std::move(frequencies)) {
        if (symbols_.size() != frequencies_.size()) throw std::invalid_argument("symbols/frequencies mismatch");
        if (symbols_.empty()) throw std::invalid_argument("empty alphabet");
        std::uint32_t total = 0;
        for (auto f : frequencies_) { if (f == 0) throw std::invalid_argument("zero frequency"); total += f; }
        if (total != table_size_) throw std::invalid_argument("frequencies must sum to table size");
        for (std::size_t i = 0; i < symbols_.size(); ++i) {
            symbol_to_index_[symbols_[i]] = static_cast<std::uint16_t>(i);
            symbol_is_present_[symbols_[i]] = true;
        }
        build_spread();
        build_decode_table();
        build_encoding_table();
    }

    PackedBitStream encode(const std::vector<std::uint8_t>& data) const {
        if (data.empty()) return {};
        std::uint32_t state = table_size_;
        const std::uint32_t original_state = state;
        PackedBitWriter writer;
        writer.reserve_bits(data.size() * 8u + 2u * table_log_);
        for (std::uint8_t sym : data) {
            if (!symbol_is_present_[sym]) throw std::invalid_argument("symbol not in alphabet");
            encode_step(state, symbol_to_index_[sym], writer);
        }
        writer.write_bits(state - table_size_, table_log_);
        writer.write_bits(original_state - table_size_, table_log_);
        PackedBitStream result;
        result.bytes = writer.bytes();
        result.bit_count = writer.bit_count();
        return result;
    }

    std::vector<std::uint8_t> decode(const std::vector<std::uint8_t>& bitstream,
                                      std::uint64_t bit_count, std::uint64_t expected_size) const {
        std::vector<std::uint8_t> result(static_cast<std::size_t>(expected_size));
        if (expected_size == 0) return result;
        if (bit_count < 2u * table_log_) throw std::invalid_argument("bitstream too small");
        PackedTailBitReader reader(bitstream, bit_count);
        const std::uint32_t original_state = reader.read_bits_from_tail(table_log_);
        std::uint32_t state = reader.read_bits_from_tail(table_log_);
        for (std::uint64_t ri = 0; ri < expected_size; ++ri) {
            if (state >= decode_table_.size()) throw std::invalid_argument("invalid decoder state");
            const DecodeEntry& entry = decode_table_[state];
            const std::uint32_t bits = reader.read_bits_from_tail(entry.nb_bits);
            state = entry.new_state + bits;
            result[static_cast<std::size_t>(expected_size - 1u - ri)] = entry.symbol;
        }
        if (state != original_state) throw std::invalid_argument("invalid final state");
        reader.assert_consumed();
        return result;
    }

private:
    void build_spread() {
        symbol_spread_.assign(table_size_, 0);
        std::uint32_t pos = 0;
        std::uint32_t step = (table_size_ >> 1u) + (table_size_ >> 3u) + 3u;
        while (std::gcd(step, table_size_) != 1u) step += 2u;
        for (std::uint16_t si = 0; si < frequencies_.size(); ++si)
            for (std::uint16_t c = 0; c < frequencies_[si]; ++c) {
                symbol_spread_[pos] = si;
                pos = (pos + step) % table_size_;
            }
    }
    void build_decode_table() {
        std::vector<std::uint32_t> next_vals(frequencies_.begin(), frequencies_.end());
        decode_table_.resize(table_size_);
        for (std::uint32_t s = 0; s < table_size_; ++s) {
            std::uint16_t si = symbol_spread_[s];
            std::uint32_t tmp = next_vals[si]++;
            std::uint8_t nb = static_cast<std::uint8_t>(table_log_ - bsr32(tmp));
            decode_table_[s] = {symbols_[si], nb, (tmp << nb) - table_size_};
        }
    }
    void build_encoding_table() {
        starts_.assign(frequencies_.size(), 0);
        std::int32_t cum = 0;
        for (std::size_t i = 0; i < frequencies_.size(); ++i) {
            starts_[i] = cum - static_cast<std::int32_t>(frequencies_[i]);
            cum += frequencies_[i];
        }
        std::vector<std::uint32_t> next_vals(frequencies_.begin(), frequencies_.end());
        encoding_table_.assign(table_size_, 0);
        for (std::uint32_t tp = 0; tp < table_size_; ++tp) {
            std::uint16_t si = symbol_spread_[tp];
            std::int32_t ei = starts_[si] + static_cast<std::int32_t>(next_vals[si]);
            encoding_table_[ei] = tp + table_size_;
            ++next_vals[si];
        }
        const std::uint32_t r = table_log_ + 1u;
        nb_values_.assign(frequencies_.size(), 0);
        for (std::size_t i = 0; i < frequencies_.size(); ++i) {
            std::uint32_t k = table_log_ - bsr32(frequencies_[i]);
            nb_values_[i] = (static_cast<std::int32_t>(k) << r) - static_cast<std::int32_t>(frequencies_[i] << k);
        }
    }
    void encode_step(std::uint32_t& state, std::uint16_t si, PackedBitWriter& w) const {
        const std::uint32_t r = table_log_ + 1u;
        const std::uint32_t nb = static_cast<std::uint32_t>(
            (static_cast<std::int64_t>(state) + nb_values_[si]) >> r);
        w.write_bits(state & ((1u << nb) - 1u), nb);
        state = encoding_table_[starts_[si] + static_cast<std::int32_t>(state >> nb)];
    }

    std::uint8_t table_log_;
    std::uint32_t table_size_;
    std::vector<std::uint8_t> symbols_;
    std::vector<std::uint16_t> frequencies_;
    std::array<std::uint16_t, kByteMaxSymbols> symbol_to_index_{};
    std::array<bool, kByteMaxSymbols> symbol_is_present_{};
    std::vector<std::uint16_t> symbol_spread_;
    std::vector<DecodeEntry> decode_table_;
    std::vector<std::uint32_t> encoding_table_;
    std::vector<std::int32_t> starts_;
    std::vector<std::int32_t> nb_values_;
};

static std::pair<std::vector<std::uint8_t>, std::vector<std::uint16_t>>
normalize_byte_frequencies(const std::array<std::uint32_t, 256>& counts,
                           std::uint64_t data_size, std::uint8_t table_log) {
    const std::uint32_t table_size = 1u << table_log;
    std::vector<std::uint8_t> symbols;
    for (std::uint32_t s = 0; s < 256; ++s)
        if (counts[s] > 0) symbols.push_back(static_cast<std::uint8_t>(s));
    if (symbols.size() > table_size) throw std::invalid_argument("table too small");
    if (symbols.size() == table_size) {
        return {symbols, std::vector<std::uint16_t>(table_size, 1)};
    }
    std::array<std::uint16_t, 256> freqs{};
    struct FP { double frac; std::uint8_t sym; };
    std::vector<FP> fps;
    std::int32_t assigned = 0;
    for (std::uint8_t sym : symbols) {
        double exact = static_cast<double>(counts[sym]) * table_size / data_size;
        std::uint16_t base = static_cast<std::uint16_t>(std::max(1, static_cast<int>(std::floor(exact))));
        freqs[sym] = base; assigned += base;
        fps.push_back({exact - std::floor(exact), sym});
    }
    std::int32_t diff = static_cast<std::int32_t>(table_size) - assigned;
    if (diff > 0) {
        std::sort(fps.begin(), fps.end(), [](const FP& a, const FP& b) {
            return a.frac != b.frac ? a.frac > b.frac : a.sym > b.sym; });
        for (std::int32_t i = 0; i < diff; ++i) ++freqs[fps[i].sym];
    } else if (diff < 0) {
        std::sort(fps.begin(), fps.end(), [](const FP& a, const FP& b) {
            return a.frac != b.frac ? a.frac < b.frac : a.sym < b.sym; });
        std::size_t idx = 0;
        for (std::int32_t i = 0; i < -diff; ++i) {
            while (idx < fps.size() && freqs[fps[idx].sym] <= 1) ++idx;
            if (idx >= fps.size()) throw std::invalid_argument("cannot normalize");
            --freqs[fps[idx].sym];
        }
    }
    // Ensure no symbol has zero frequency after normalization
    for (std::uint8_t sym : symbols) {
        if (freqs[sym] == 0) {
            // Find the symbol with highest frequency to steal from
            std::uint8_t donor = symbols[0];
            for (std::uint8_t s : symbols)
                if (freqs[s] > freqs[donor]) donor = s;
            if (freqs[donor] <= 1) throw std::invalid_argument("cannot normalize: all frequencies are 1");
            --freqs[donor];
            freqs[sym] = 1;
        }
    }
    std::vector<std::uint16_t> out; out.reserve(symbols.size());
    for (std::uint8_t sym : symbols) out.push_back(freqs[sym]);
    return {symbols, out};
}

static py::bytes tans_encode_bytes_cpp(py::bytes input_bytes, std::uint32_t table_log = 8) {
    std::string_view sv(input_bytes);
    if (table_log < 5 || table_log > 15) throw std::invalid_argument("table_log must be 5-15");
    std::vector<std::uint8_t> packed;
    {
        py::gil_scoped_release release;
        const auto* data = reinterpret_cast<const std::uint8_t*>(sv.data());
        std::uint64_t n = sv.size();
        std::array<std::uint32_t, 256> counts{};
        for (std::uint64_t i = 0; i < n; ++i) counts[data[i]]++;
        auto norm = normalize_byte_frequencies(counts, n, static_cast<std::uint8_t>(table_log));
        ByteTansCoder coder(static_cast<std::uint8_t>(table_log), norm.first, norm.second);
        std::vector<std::uint8_t> input_vec(data, data + n);
        auto stream = coder.encode(input_vec);
        packed.reserve(26 + norm.first.size() * 3 + stream.bytes.size());
        packed.insert(packed.end(), kByteTansMagic, kByteTansMagic + 4);
        append_u64_le(packed, n);
        packed.push_back(static_cast<std::uint8_t>(table_log));
        append_u16_le(packed, static_cast<std::uint16_t>(norm.first.size()));
        append_u64_le(packed, stream.bit_count);
        append_u32_le(packed, static_cast<std::uint32_t>(stream.bytes.size()));
        for (std::size_t i = 0; i < norm.first.size(); ++i) {
            packed.push_back(norm.first[i]);
            append_u16_le(packed, norm.second[i]);
        }
        packed.insert(packed.end(), stream.bytes.begin(), stream.bytes.end());
    }
    return py::bytes(reinterpret_cast<const char*>(packed.data()), packed.size());
}

static py::bytes tans_decode_bytes_cpp(py::bytes encoded_bytes) {
    std::string_view sv(encoded_bytes);
    std::vector<std::uint8_t> output;
    {
        py::gil_scoped_release release;
        if (sv.size() < 26) throw std::invalid_argument("payload too small");
        const auto* p = reinterpret_cast<const std::uint8_t*>(sv.data());
        if (p[0]!='B'||p[1]!='T'||p[2]!='A'||p[3]!='N') throw std::invalid_argument("bad magic");
        std::uint64_t orig_size = read_u64_le(p + 4);
        std::uint8_t tlog = p[12];
        std::uint16_t sc = read_u16_le(p + 13);
        std::uint64_t bit_count = read_u64_le(p + 15);
        std::uint32_t ss = read_u32_le(p + 23);
        std::vector<std::uint8_t> symbols; symbols.reserve(sc);
        std::vector<std::uint16_t> freqs; freqs.reserve(sc);
        std::size_t off = 27;
        for (std::uint16_t i = 0; i < sc; ++i) {
            symbols.push_back(p[off++]);
            freqs.push_back(read_u16_le(p + off)); off += 2;
        }
        std::vector<std::uint8_t> bitstream(p + off, p + off + ss);
        if (orig_size == 0) { output.clear(); }
        else {
            ByteTansCoder coder(tlog, std::move(symbols), std::move(freqs));
            output = coder.decode(bitstream, bit_count, orig_size);
        }
    }
    return py::bytes(reinterpret_cast<const char*>(output.data()), output.size());
}

}  // namespace

PYBIND11_MODULE(drotl1fmd_tans_cpp_codec, module) {
    module.doc() = "High-performance C++ tANS codec";
    module.def("encode_drotl1fmd_tans_f32", &encode_drotl1fmd_tans_f32_cpp,
               py::arg("reference_bytes"), py::arg("target_bytes"),
               py::arg("table_log") = kDefaultTableLog);
    module.def("decode_drotl1fmd_tans_f32", &decode_drotl1fmd_tans_f32_cpp,
               py::arg("encoded_bytes"), py::arg("reference_bytes"));
    module.def("encode_pcdelta_tans_f32", &encode_pcdelta_tans_f32_cpp,
               py::arg("reference_bytes"), py::arg("target_bytes"),
               py::arg("table_log") = kDefaultTableLog);
    module.def("decode_pcdelta_tans_f32", &decode_pcdelta_tans_f32_cpp,
               py::arg("encoded_bytes"), py::arg("reference_bytes"));
    module.def("tans_encode", &tans_encode_bytes_cpp,
               py::arg("data"), py::arg("table_log") = 8);
    module.def("tans_decode", &tans_decode_bytes_cpp,
               py::arg("encoded"));
}
