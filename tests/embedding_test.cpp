#include "model/embedding.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
using mini_llm::model::Embedding;

constexpr Embedding::TokenId kInvalidTokenId = -1;

void expect_true(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "[FAIL] " << message << '\n';
        std::exit(1);
    }
}

std::filesystem::path make_temp_path(const std::string& prefix)
{
    namespace fs = std::filesystem;

    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << prefix << '-' << now << '-' << std::rand() << ".json";
    return fs::temp_directory_path() / "qwen_cpp_embedding_tests" / oss.str();
}

class ScopedTempFile
{
public:
    explicit ScopedTempFile(const std::string& content)
    {
        namespace fs = std::filesystem;

        const fs::path dir = fs::temp_directory_path() / "qwen_cpp_embedding_tests";
        fs::create_directories(dir);

        path_ = make_temp_path("vocab");

        std::ofstream ofs(path_, std::ios::binary);
        if (!ofs.is_open())
        {
            throw std::runtime_error("failed to create temp vocab file");
        }
        ofs << content;
        ofs.flush();
        if (!ofs)
        {
            throw std::runtime_error("failed to write temp vocab file");
        }
    }

    ~ScopedTempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void test_basic_load_and_mapping_queries()
{
    Embedding embedding;
    expect_true(embedding.empty(), "new embedding should be empty");
    expect_true(embedding.size() == 0, "new embedding size should be 0");

    const ScopedTempFile file(R"({"<unk>":0,"hello":"2","world":5})");
    const bool loaded = embedding.load_vocab(file.path().string());

    expect_true(loaded, "load_vocab should return true for valid vocab");
    expect_true(!embedding.empty(), "embedding should not be empty after successful load");
    expect_true(embedding.size() == 3, "size should equal unique token count");

    expect_true(embedding.contains_token("<unk>"), "contains_token should find existing token");
    expect_true(!embedding.contains_token("missing"), "contains_token should fail for missing token");

    expect_true(embedding.contains_id(0), "contains_id should accept valid id 0");
    expect_true(embedding.contains_id(2), "contains_id should accept valid id 2");
    expect_true(embedding.contains_id(5), "contains_id should accept valid id 5");
    expect_true(!embedding.contains_id(1), "contains_id should reject sparse hole id");
    expect_true(!embedding.contains_id(-1), "contains_id should reject negative id");
    expect_true(!embedding.contains_id(6), "contains_id should reject out-of-range id");

    expect_true(embedding.token_to_id("hello") == 2, "token_to_id should map token to configured id");
    expect_true(embedding.token_to_id("world") == 5, "token_to_id should map second token");
    expect_true(embedding.token_to_id("missing") == kInvalidTokenId,
                "token_to_id should return invalid id when token and UNK are absent");

    expect_true(embedding.id_to_token(0) == "<unk>", "id_to_token should return token for id 0");
    expect_true(embedding.id_to_token(2) == "hello", "id_to_token should return token for sparse valid id");
    expect_true(embedding.id_to_token(1).empty(), "id_to_token should return empty string for sparse hole id");
    expect_true(embedding.id_to_token(-1).empty(), "id_to_token should return empty string for negative id");

    const auto& vocab = embedding.vocabulary();
    expect_true(vocab.size() == 6, "vocabulary vector should be sized by max_id + 1");
    expect_true(vocab[0] == "<unk>", "vocabulary slot 0 should be <unk>");
    expect_true(vocab[1].empty(), "vocabulary slot 1 should be empty hole");
    expect_true(vocab[2] == "hello", "vocabulary slot 2 should be hello");
    expect_true(vocab[5] == "world", "vocabulary slot 5 should be world");
}

void test_special_tokens_getters_and_fallback_behaviour()
{
    Embedding embedding;
    const ScopedTempFile file(R"({"<unk>":0,"<pad>":1,"<bos>":2,"<eos>":3,"known":4})");
    static_cast<void>(embedding.load_vocab(file.path().string()));

    embedding.set_unk_token("<unk>");
    embedding.set_pad_token("<pad>");
    embedding.set_bos_token("<bos>");
    embedding.set_eos_token("<eos>");

    expect_true(embedding.unk_token() == "<unk>", "unk_token getter should return configured value");
    expect_true(embedding.pad_token() == "<pad>", "pad_token getter should return configured value");
    expect_true(embedding.bos_token() == "<bos>", "bos_token getter should return configured value");
    expect_true(embedding.eos_token() == "<eos>", "eos_token getter should return configured value");

    expect_true(embedding.unk_token_id() == 0, "unk_token_id should resolve to vocab id");
    expect_true(embedding.pad_token_id() == 1, "pad_token_id should resolve to vocab id");
    expect_true(embedding.bos_token_id() == 2, "bos_token_id should resolve to vocab id");
    expect_true(embedding.eos_token_id() == 3, "eos_token_id should resolve to vocab id");

    expect_true(embedding.token_to_id("known") == 4, "token_to_id should return direct id for known token");
    expect_true(embedding.token_to_id("missing") == 0,
                "token_to_id should fallback to UNK id for unknown token when UNK exists");

    embedding.set_unk_token("not_in_vocab");
    expect_true(embedding.unk_token_id() == kInvalidTokenId,
                "unk_token_id should return invalid id if configured token not in vocab");
    expect_true(embedding.token_to_id("missing") == kInvalidTokenId,
                "token_to_id should return invalid id if UNK token configured but not in vocab");
}

void test_load_vocab_missing_file_returns_false()
{
    Embedding embedding;
    const std::filesystem::path missing = make_temp_path("missing-vocab");

    const bool loaded = embedding.load_vocab(missing.string());
    expect_true(!loaded, "load_vocab should return false for missing file path");
    expect_true(embedding.empty(), "embedding should remain empty after missing-file load");
}

void test_load_vocab_parse_and_schema_errors()
{
    {
        Embedding embedding;
        const ScopedTempFile file("{invalid-json");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::runtime_error&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw runtime_error on malformed JSON");
    }

    {
        Embedding embedding;
        const ScopedTempFile file("[1,2,3]");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw invalid_argument when JSON root is not object");
    }

    {
        Embedding embedding;
        const ScopedTempFile file(R"({"":1})");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw invalid_argument on empty token text");
    }
}

void test_load_vocab_invalid_id_value_errors()
{
    {
        Embedding embedding;
        const ScopedTempFile file(R"({"hello":"12a"})");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw invalid_argument for non-integer id string");
    }

    {
        Embedding embedding;
        const ScopedTempFile file(R"({"hello":true})");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw invalid_argument for unsupported id value type");
    }

    {
        Embedding embedding;
        const ScopedTempFile file(R"({"hello":-1})");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::out_of_range&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw out_of_range for negative id");
    }

    {
        Embedding embedding;
        const ScopedTempFile file(R"({"hello":2147483648})");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::out_of_range&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw out_of_range for id beyond TokenId max");
    }
}

void test_load_vocab_duplicate_token_and_duplicate_id_errors()
{
    {
        Embedding embedding;
        // simdjson 会保留对象迭代中的重复 key，便于覆盖 Duplicate token 分支。
        const ScopedTempFile file(R"({"dup":1,"dup":2})");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw invalid_argument for duplicate token key");
    }

    {
        Embedding embedding;
        const ScopedTempFile file(R"({"a":1,"b":1})");

        bool thrown = false;
        try
        {
            static_cast<void>(embedding.load_vocab(file.path().string()));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "load_vocab should throw invalid_argument for duplicate token id");
    }
}

void test_failed_load_does_not_mutate_existing_state()
{
    Embedding embedding;

    const ScopedTempFile valid_file(R"({"ok":0,"x":3})");
    expect_true(embedding.load_vocab(valid_file.path().string()), "initial valid load should succeed");

    expect_true(embedding.size() == 2, "precondition: size should be 2 after first successful load");
    expect_true(embedding.id_to_token(3) == "x", "precondition: id 3 should map to x");

    const ScopedTempFile invalid_file(R"({"bad":"9x"})");

    bool thrown = false;
    try
    {
        static_cast<void>(embedding.load_vocab(invalid_file.path().string()));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }
    expect_true(thrown, "invalid load should throw");

    expect_true(embedding.size() == 2, "failed load should keep previous size unchanged");
    expect_true(embedding.token_to_id("ok") == 0, "failed load should keep previous token mapping");
    expect_true(embedding.id_to_token(3) == "x", "failed load should keep previous id mapping");
}
} // namespace

int main()
{
    test_basic_load_and_mapping_queries();
    test_special_tokens_getters_and_fallback_behaviour();
    test_load_vocab_missing_file_returns_false();
    test_load_vocab_parse_and_schema_errors();
    test_load_vocab_invalid_id_value_errors();
    test_load_vocab_duplicate_token_and_duplicate_id_errors();
    test_failed_load_does_not_mutate_existing_state();

    std::cout << "[PASS] Embedding tests passed.\n";
    return 0;
}
