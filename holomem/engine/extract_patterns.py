"""Auto-extraction patterns, copied by tools/sync_engine.py out of the provider (__init__.py) of itpartypattaya/hermes-plugin-holographic v0.24.0:
the source text of its _EXTRACT_CATEGORIES assignment, exactly. Do not edit: change the plugin."""

import re

_EXTRACT_CATEGORIES = (
    # "want"/"need" are deliberately NOT here: they state a one-off intent ("I want to ask you about the
    # date", "I need the 2023 production"), and a user who pastes a model's English draft to translate it
    # sends that draft as role="user", so the model's own reasoning sentences got stored as user_pref facts.
    # prefer/like/love/use keep the durable-preference signal without it.
    # The same for "хочу" in Russian (0.17.6): "я хочу уточнить дату" is an intent, not a preference.
    ([re.compile(r'\bI\s+(?:prefer|like|love|use)\s+\S', re.IGNORECASE),
      re.compile(r'\bmy\s+(?:favorite|favourite|preferred|default)\s+\w+\s+is\s+\S', re.IGNORECASE),
      re.compile(r'\bI\s+(?:always|never|usually)\s+\S', re.IGNORECASE),
      re.compile(r'(?:^|\s)я\s+(?:предпочитаю|люблю|использую|пользуюсь|обычно|всегда|никогда|не\s+ем|не\s+пью)\s+\S', re.IGNORECASE),
      re.compile(r'(?:^|\s)мо[йяеи]\s+любим\w+\s+\S', re.IGNORECASE),
      # Chinese, Traditional and Simplified (after itpartypattaya/hermes-plugin-holographic#2 by au61987ai-code). The
      # sentence must open with the preference — "I" or no subject — so "你喜歡什麼" (what do YOU like) and "他喜歡吃辣"
      # (HE likes spicy food) are not the user's; "不用" (no need) and "我覺得 / 以我來看" (passing opinions) are left out.
      re.compile(r'^(?:我|本人)?(?:也|還是|还是|一直|通常|比較|比较|較|较|更)?'
                 r'(?:偏好|習慣|习惯|喜歡|喜欢|愛用|爱用|傾向|倾向|會先?選用|会先?选用)[^。；！？?\n]{2,40}'),
      re.compile(r'^(?:我|本人)(?:一律|向來|向来|總是|总是|從不|从不|不習慣|不习惯|不接受|不再)[^。；！？?\n]{2,40}')],
     "user_pref"),
    ([re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+\S', re.IGNORECASE),
      re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+\S', re.IGNORECASE),
      re.compile(r'(?:^|\s)мы\s+(?:решили|договорились|выбрали)\s+\S', re.IGNORECASE),
      re.compile(r'(?:^|\s)в\s+проекте\s+(?:используем|используется|нужен|нужна|нужно)\s+\S', re.IGNORECASE),
      # Chinese: an explicit deciding subject ("你必須先備份" is an order to the assistant, not a decision), or the
      # project / its rules as the subject, as "the project uses …" is in English.
      re.compile(r'^(?:我們|我们|大家|團隊|团队|雙方|双方)(?:已經|已经|最後|最后|一致)?'
                 r'(?:決定|决定|決議|决议|達成共識|达成共识|商定|約定好|约定好)[^。；！？?\n]{2,60}'),
      re.compile(r'^(?:這個|这个|本|我們的|我们的)?(?:專案|项目|項目)(?:裡|里|中)?'
                 r'(?:使用|採用|采用|需要|必須用|必须用|一律用|統一用|统一用)[^。；！？?\n]{2,40}'),
      re.compile(r'^(?:這個|这个|本|我們的|我们的)?(?:專案|项目|項目)?的?'
                 r'(?:命名規則|命名规则|命名規範|命名规范|規範|规范|慣例|惯例)(?:是|為|为|採用|采用)[^。；！？?\n]{2,40}')],
     "project"),
)
