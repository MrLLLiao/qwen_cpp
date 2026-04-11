#include "service/http_server.h"

namespace mini_llm::service {

bool HttpServer::start(const std::string& host, int port) {
    // TODO: 初始化 HTTP server、注册中间件与路由
    (void)host;
    (void)port;
    return false;
}

void HttpServer::stop() {
    // TODO: 实现优雅停机与会话资源清理
}

} // namespace mini_llm::service
