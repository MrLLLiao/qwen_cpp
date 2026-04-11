#pragma once

#include <string>

namespace mini_llm::service {

class HttpServer {
public:
    // TODO: 绑定 /health /models /generate /chat 路由
    bool start(const std::string& host, int port);
    void stop();
};

} // namespace mini_llm::service
