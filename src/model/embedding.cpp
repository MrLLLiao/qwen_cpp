#include "model/embedding.h"

#include <algorithm>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <limits>

using namespace mini_llm::model;

namespace
{
constexpr Embedding::TokenId kInvalidTokenId = -1;
}

bool Embedding::load_vocab(const std::string& vocab_path)
{
    namespace fs = std::filesystem;

    const fs::path path(vocab_path);
    if (!fs::exists(path) || !fs::is_regular_file(path))
    {
        return false;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Embedding::load_vocab: Can't open file: " + path.string());
    }

    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    if (doc.HasParseError())
    {
        throw std::runtime_error(
            std::string("Embedding::load_vocab: Parsing error at offset ") +
            std::to_string(doc.GetErrorOffset()));
    }

    if (!doc.IsObject())
    {
        throw std::invalid_argument("Embedding::load_vocab: JSON root must be an object: {\"token\": id}");
    }

    std::unordered_map<std::string, TokenId> new_token_to_id;
    new_token_to_id.reserve(doc.MemberCount());

    TokenId max_id = -1;

    for (auto& [name, value] : doc.GetObject())
    {
        const std::string token = name.GetString();
        if (token.empty())
        {
            throw std::invalid_argument("Embedding::load_vocab: Token text must not be empty.");
        }

        std::int64_t parsed_id = 0;
        if (value.IsString())
        {
            try
            {
                std::size_t processed = 0;
                const std::string id_str = value.GetString();
                parsed_id = std::stoll(id_str, &processed);
                if (processed != id_str.size())
                {
                    throw std::invalid_argument("extra characters");
                }
            }
            catch (const std::exception&)
            {
                throw std::invalid_argument("Embedding::load_vocab: Invalid id string for token: " + token);
            }
        }
        else if (value.IsInt64())
        {
            parsed_id = value.GetInt64();
        }
        else
        {
            throw std::invalid_argument("Embedding::load_vocab: Token id must be integer/string integer. token=" + token);
        }

        if (parsed_id < 0 || parsed_id > std::numeric_limits<TokenId>::max())
        {
            throw std::out_of_range("Embedding::load_vocab: Token id out of range for token: " + token);
        }

        const auto token_id = static_cast<TokenId>(parsed_id);
        if (new_token_to_id.contains(token))
        {
            throw std::invalid_argument("Embedding::load_vocab: Duplicate token: " + token);
        }

        new_token_to_id.emplace(token, token_id);
        max_id = std::max(max_id, token_id);
    }

    std::vector<std::string> new_id_to_token;
    if (max_id >= 0)
    {
        new_id_to_token.resize(static_cast<std::size_t>(max_id) + 1);
    }

    for (const auto& [token, token_id] : new_token_to_id)
    {
        auto& slot = new_id_to_token[static_cast<std::size_t>(token_id)];
        if (!slot.empty())
        {
            throw std::invalid_argument("Embedding::load_vocab: Duplicate token id: " + std::to_string(token_id));
        }
        slot = token;
    }

    token_to_id_table_ = std::move(new_token_to_id);
    id_to_token_table_ = std::move(new_id_to_token);

    return true;
}

bool Embedding::empty() const
{
    return token_to_id_table_.empty() || id_to_token_table_.empty();
}

std::size_t Embedding::size() const
{
    return token_to_id_table_.size();
}

bool Embedding::contains_token(const std::string& token) const
{
    return token_to_id_table_.contains(token);
}

bool Embedding::contains_id(TokenId id) const
{
    return is_valid_id(id);
}

Embedding::TokenId Embedding::token_to_id(const std::string& token) const
{
    return find_token_id_or_default(token);
}

const std::string& Embedding::id_to_token(TokenId id) const
{
    if (is_valid_id(id))
    {
        return id_to_token_table_.at(static_cast<std::size_t>(id));
    }
    return empty_token();
}

void Embedding::set_unk_token(const std::string& token)
{
    unk_token_ = token;
}

void Embedding::set_pad_token(const std::string& token)
{
    pad_token_ = token;
}

void Embedding::set_bos_token(const std::string& token)
{
    bos_token_ = token;
}

void Embedding::set_eos_token(const std::string& token)
{
    eos_token_ = token;
}

const std::string& Embedding::unk_token() const
{
    return unk_token_;
}

const std::string& Embedding::pad_token() const
{
    return pad_token_;
}

const std::string& Embedding::bos_token() const
{
    return bos_token_;
}

const std::string& Embedding::eos_token() const
{
    return eos_token_;
}

Embedding::TokenId Embedding::unk_token_id() const
{
    const auto it = token_to_id_table_.find(unk_token_);
    if (it != token_to_id_table_.end())
    {
        return it->second;
    }
    return kInvalidTokenId;
}

Embedding::TokenId Embedding::pad_token_id() const
{
    const auto it = token_to_id_table_.find(pad_token_);
    if (it != token_to_id_table_.end())
    {
        return it->second;
    }
    return kInvalidTokenId;
}

Embedding::TokenId Embedding::bos_token_id() const
{
    const auto it = token_to_id_table_.find(bos_token_);
    if (it != token_to_id_table_.end())
    {
        return it->second;
    }
    return kInvalidTokenId;
}

Embedding::TokenId Embedding::eos_token_id() const
{
    const auto it = token_to_id_table_.find(eos_token_);
    if (it != token_to_id_table_.end())
    {
        return it->second;
    }
    return kInvalidTokenId;
}

const std::string& Embedding::empty_token()
{
    static const std::string EmptyString{};
    return EmptyString;
}

const std::vector<std::string>& Embedding::vocabulary() const
{
    return id_to_token_table_;
}

Embedding::TokenId Embedding::find_token_id_or_default(const std::string& token) const
{
    const auto token_it = token_to_id_table_.find(token);
    if (token_it != token_to_id_table_.end())
    {
        return token_it->second;
    }

    if (!unk_token_.empty())
    {
        const auto unk_it = token_to_id_table_.find(unk_token_);
        if (unk_it != token_to_id_table_.end())
        {
            return unk_it->second;
        }
    }

    return kInvalidTokenId;
}

bool Embedding::is_valid_id(TokenId id) const
{
    if (id < 0)
    {
        return false;
    }

    const auto index = static_cast<std::size_t>(id);
    if (index >= id_to_token_table_.size())
    {
        return false;
    }

    return !id_to_token_table_[index].empty();
}