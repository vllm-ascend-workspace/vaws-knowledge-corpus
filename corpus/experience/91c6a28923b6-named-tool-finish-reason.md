# named tool 请求被错误的 finish_reason 断言判成失败

2026 年 7 月 30 日，Kimi K3 的四 DP 功能验收脚本在工具调用项中要求 `finish_reason == "tool_calls"`。继续核对脚本构造的请求和当时 vLLM serving 源码后，发现测试条件与请求模式不一致：请求显式指定了函数名称，属于 named tool，不能沿用自动选择工具时的结束原因断言。

当时请求的关键部分如下，模型名和其余服务参数省略：

```json
{
  "messages": [{"role": "user", "content": "Use get_weather to check the weather in Paris."}],
  "tool_choice": {
    "type": "function",
    "function": {"name": "get_weather"}
  },
  "temperature": 0,
  "stream": false
}
```

同一 payload 提供了 `get_weather` 的函数 schema，必需参数为字符串 `city`。验收脚本先检查 `finish_reason` 和工具数量，再检查函数名、将 arguments 解析为 JSON、确认 `city == "Paris"`。所以即使工具名和参数完全正确，第一条不适用的 finish reason 断言也会提前令用例失败。

源码核对覆盖当时 v0.26 和当时 main 的 streaming／non-streaming 分支。非流式路径仅在自动工具解析成功，或 `tool_choice == "required"` 且底层正常停止时，将 finish reason 映射为 `tool_calls`；named-tool 对象不满足这个字符串判断，保留底层停止原因，无值时回落为 `stop`。流式路径同样通过是否存在 `tool_choice_function_name` 区分 named 与自动选择，不是只要消息里出现 tool_calls 就统一改写结束原因。

因此该脚本的最小纠正是让结束原因断言与所请求的模式一致：本次 named-tool 的正常结束应接受 `stop`；若明确要验证 auto／required，则应另构造对应请求。工具数量、函数名、arguments 可解析性和 `city` 的检查仍需保留，不能通过删除全部工具内容断言来让测试“变绿”。如果响应因输出预算耗尽而返回 `length`，也不能把它伪装成正常 named-tool 完成。

本次完成的是脚本与公共 serving 分支的静态对照，未取得修正断言后的完整四 DP 工具调用重跑结果。因此它能确定一个验收脚本误判点，不能代替对模型生成参数、服务解析和真实响应的后续验收，也不声称现在所有版本都使用相同映射规则。
