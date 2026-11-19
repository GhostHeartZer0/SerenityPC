import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
import re


class TestStopTokenAndPromptSafety(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_closers_patterns_do_not_contain_turn_stop_tokens(self):
        """Ensures stop tokens (<|im_end|>, <|endoftext|>, <｜end▁of▁sentence｜>, <|end|>) are not treated as thought closers."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        # Check source of main.py around closers_patterns
        with open("main.py", "r", encoding="utf-8") as f:
            src = f.read()

        m = re.search(r'closers_patterns\s*=\s*\[([\s\S]*?)\]', src)
        self.assertTrue(m, "closers_patterns should be defined in main.py")
        closers_content = m.group(1)
        self.assertNotIn("<|im_end|>", closers_content)
        self.assertNotIn("<|eom|>", closers_content)
        self.assertNotIn("<｜end▁of▁sentence｜>", closers_content)
        self.assertNotIn("<|end|>", closers_content)

    def test_direct_answer_with_im_end_does_not_trigger_synthesis(self):
        """When Qwen ends with <|im_end|>, it must be recognized as final answer, not empty thoughts."""
        response_text = "Here is the exact code you requested:\ndef add(a, b):\n    return a + b\n<|im_end|>"
        # Simulate scout & split logic in main.py
        closers_patterns = [
            r'<\/think>', r'<\/thought>', r'<\/\|think\|>', r'<\/\|thought\|>',
            r'\[\/THOUGHT\]', r'\[\/thought\]', r'\[\/thinking\]', r'\[\/reasoning\]', r'\[\/DRAFT\]',
            r'<\|channel>text', r'<\|channel>assistant', r'<\|channel\|>final', r'<\|channel\|>text', r'<\|channel\|>assistant',
            r'<channel\|>', r'<\/channel\|>',
            r'[\u27e7⟧]\s*<\/think>', r'[\u27e7⟧]',
            r'<\|start\|>assistant\s+to=user(?:<\|message\|>)?', r'to=user<\|message\|>',
            r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Answer|Execution complete)[\s:]+'
        ]
        all_splits = []
        for tag_pattern in closers_patterns:
            for m in re.finditer(tag_pattern, response_text, re.IGNORECASE):
                all_splits.append(m.end())

        best_split = -1
        if all_splits:
            for split in all_splits:
                remaining = response_text[split:].strip()
                if re.search(r'<think>|<thought>|\[DRAFT\]|\[thinking\]|\[thought\]|\[reasoning\]|\[THOUGHT\]', remaining, re.IGNORECASE):
                    continue
                best_split = split
                break
            if best_split == -1:
                best_split = all_splits[-1]

        if best_split != -1:
            think_log = response_text[:best_split].strip()
            final_answer = response_text[best_split:].strip()
        else:
            has_thought_openers = bool(re.search(r'<think>|<thought>|\[DRAFT\]|\[thinking\]|\[thought\]|\[reasoning\]|\[THOUGHT\]', response_text, re.IGNORECASE))
            if has_thought_openers:
                think_log = response_text
                final_answer = ""
            else:
                think_log = ""
                final_answer = response_text.strip()

        # Clean tags
        tag_clean_pattern = r'(?i)<\|im_end\|?>|<\/think>'
        final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()

        self.assertEqual(think_log, "")
        self.assertIn("def add(a, b):", final_answer)
        self.assertNotIn("<|im_end|>", final_answer)
        # Crucial check: think_log and not final_answer must be FALSE so synthesis does not fire
        synthesis_triggered = bool(think_log and not final_answer)
        self.assertFalse(synthesis_triggered, "Stop token should not trigger synthesis fallback restart")

    def test_tool_loop_suppresses_missing_required_args(self):
        """Calling tools like send_peer_message with empty args must be suppressed without restarting inference."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app.config = {}
        app.tool_registry = MagicMock()
        app.stop_process = MagicMock()
        app.stop_process.is_set.return_value = False
        app.process_queue = MagicMock()
        app.active_persona_level = 2

        # Test tool loop guard directly
        full_resp = "Here is some info about peer messages: send_peer_message()"
        prompt_str = "Explain peer messages"
        params = {}

        result = ChatbotApp._run_tool_loop(app, full_resp, prompt_str, params)
        self.assertEqual(result, full_resp)
        # execute should not be called because args are empty for send_peer_message
        app.tool_registry.execute.assert_not_called()

    def test_run_blocking_inference_defaults_stream_to_ui_false(self):
        """_run_blocking_inference should not stream intermediate synthesis to chat UI unless requested."""
        import inspect
        from main import ChatbotApp
        sig = inspect.signature(ChatbotApp._run_blocking_inference)
        self.assertIn("stream_to_ui", sig.parameters)
        self.assertFalse(sig.parameters["stream_to_ui"].default)

    def test_chat_mark_protects_user_prompt(self):
        """user_prompt_end mark prevents deletion from wiping user prompt in chat_history."""
        hist = tk.Text(self.root)
        hist.pack()

        # Insert user prompt
        hist.insert(tk.END, "\nPrompt: ", ("user_lead",))
        start_idx = hist.index(tk.END + "-1c")
        hist.insert(tk.END, "Critical user prompt with hours of work\n", ("user",))
        end_idx = hist.index(tk.END + "-1c")

        hist.mark_set("user_prompt_start", start_idx)
        hist.mark_gravity("user_prompt_start", tk.LEFT)
        hist.mark_set("user_prompt_end", end_idx)
        hist.mark_gravity("user_prompt_end", tk.LEFT)
        hist.mark_set("response_start", end_idx)
        hist.mark_gravity("response_start", tk.LEFT)

        # Insert AI stream
        hist.insert(tk.END, "\n\nSerenity: Partial answer text", ("ai",))

        # Attempt rogue deletion from before user_prompt_end
        rogue_start = "1.0"
        if "user_prompt_end" in hist.mark_names() and hist.compare(rogue_start, "<", "user_prompt_end"):
            safe_start = hist.index("user_prompt_end")
        else:
            safe_start = rogue_start

        # Deletion using safe_start
        hist.delete(safe_start, tk.END)

        # Verify prompt is completely preserved
        content = hist.get("1.0", tk.END)
        self.assertIn("Critical user prompt with hours of work", content)
        self.assertNotIn("Partial answer text", content)

    def test_halt_process_commits_unsaved_turn(self):
        """Halting commits the prompt and partial response to self.messages so user never loses history."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app.state = {"running": True, "was_halted": False}
        app.config = {"halt_behavior_mode": "off", "history_autosave_mode": "End"}
        app.stop_process = MagicMock()
        app.last_user_message = "Crucial prompt before halt"
        app.messages = []
        app.text_buffer = ""
        app.thought_stream_buffer = ""
        app.thinking_display = None
        app.chat_history = tk.Text(self.root)
        app.chat_history.insert("1.0", "Prompt: Crucial prompt before halt\nSerenity: Partial generation before halt")
        app.chat_history.mark_set("response_start", "2.0")
        app._get_persona_label = MagicMock(return_value="Serenity")
        app.save_history = MagicMock()

        ChatbotApp.halt_process(app)

        self.assertTrue(app.stop_process.set.called)
        self.assertEqual(len(app.messages), 2)
        self.assertEqual(app.messages[0]["role"], "user")
        self.assertEqual(app.messages[0]["content"], "Crucial prompt before halt")
        self.assertEqual(app.messages[1]["role"], "assistant")
        self.assertTrue(app.save_history.called)

    def test_restore_last_prompt_shortcut(self):
        """_restore_last_prompt correctly restores the last submitted prompt into user_input."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app.user_input = tk.Text(self.root, undo=True)
        app._last_submitted_prompt = "My previous lengthy query"

        res = ChatbotApp._restore_last_prompt(app)
        self.assertEqual(res, 'break')
        self.assertEqual(app.user_input.get("1.0", tk.END).strip(), "My previous lengthy query")

    def test_up_arrow_with_numlock_does_not_revert_prompt(self):
        """Pressing Up with Windows Num Lock (state 0x0008) must not wipe typed progress."""
        from types import SimpleNamespace
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app._slash_popup = None
        app.user_input = tk.Text(self.root, undo=True)
        app.user_input.insert("1.0", "Line 1\nLine 2 in progress")
        app._last_submitted_prompt = "Old previous prompt"

        event = SimpleNamespace(keysym="Up", state=0x0008)
        res = ChatbotApp._handle_input_key(app, event)
        self.assertIsNone(res)
        self.assertEqual(app.user_input.get("1.0", "end-1c"), "Line 1\nLine 2 in progress")

    def test_restore_last_prompt_preserves_draft_and_only_if_empty(self):
        """only_if_empty=True skips overwrite; Alt+Up stashes draft recoverable via _restore_prompt_draft."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app.user_input = tk.Text(self.root, undo=True)
        app.user_input.insert("1.0", "Unsent draft progress")
        app._last_submitted_prompt = "Previous prompt"

        # Ctrl+Up (only_if_empty=True) must not overwrite non-empty draft
        self.assertIsNone(ChatbotApp._restore_last_prompt(app, only_if_empty=True))
        self.assertEqual(app.user_input.get("1.0", "end-1c"), "Unsent draft progress")

        # Alt+Up restores previous prompt but stashes draft
        self.assertEqual(ChatbotApp._restore_last_prompt(app), 'break')
        self.assertEqual(app.user_input.get("1.0", "end-1c"), "Previous prompt")
        self.assertEqual(app._stashed_prompt_draft, "Unsent draft progress")

        # Alt+Down restores stashed draft
        self.assertEqual(ChatbotApp._restore_prompt_draft(app), 'break')
        self.assertEqual(app.user_input.get("1.0", "end-1c"), "Unsent draft progress")

    def test_media_gen_mode_cycling_and_colors(self):
        """Media generation button cycles off -> image -> gif -> video -> off and updates colors/labels."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app.config = {"media_gen_mode": "off"}
        app.media_gen_button = MagicMock()
        app._get_media_gen_label = lambda: ChatbotApp._get_media_gen_label(app)
        app._get_media_gen_color = lambda: ChatbotApp._get_media_gen_color(app)

        self.assertEqual(ChatbotApp._get_media_gen_label(app), "🎨 Gen:\nOFF")
        self.assertEqual(ChatbotApp._get_media_gen_color(app), "#888888")

        ChatbotApp.toggle_media_gen_mode(app)
        self.assertEqual(app.config["media_gen_mode"], "image")
        self.assertEqual(ChatbotApp._get_media_gen_label(app), "🎨 Gen:\nIMG")
        self.assertEqual(ChatbotApp._get_media_gen_color(app), "#00ffcc")

        ChatbotApp.toggle_media_gen_mode(app)
        self.assertEqual(app.config["media_gen_mode"], "gif")
        self.assertEqual(ChatbotApp._get_media_gen_label(app), "🎞️ Gen:\nGIF")
        self.assertEqual(ChatbotApp._get_media_gen_color(app), "#ff77ff")

        ChatbotApp.toggle_media_gen_mode(app)
        self.assertEqual(app.config["media_gen_mode"], "video")
        self.assertEqual(ChatbotApp._get_media_gen_label(app), "🎬 Gen:\nVID")
        self.assertEqual(ChatbotApp._get_media_gen_color(app), "#00ff66")

        ChatbotApp.toggle_media_gen_mode(app)
        self.assertEqual(app.config["media_gen_mode"], "off")

    def test_generate_video_tool_loop_and_validation(self):
        """generate_video is recognized in _run_tool_loop and suppressed when prompt is empty."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app.config = {}
        app.tool_registry = MagicMock()
        app.stop_process = MagicMock()
        app.stop_process.is_set.return_value = False
        app.process_queue = MagicMock()
        app.active_persona_level = 2

        # 1. Empty args should be suppressed
        empty_resp = "generate_video()"
        res1 = ChatbotApp._run_tool_loop(app, empty_resp, "make video", {})
        self.assertEqual(res1, empty_resp)
        app.tool_registry.execute.assert_not_called()

        # 2. Valid prompt arg should invoke tool_registry.execute("generate_video", {"prompt": "neon city"})
        app._run_tool_loop = lambda *a, **kw: ChatbotApp._run_tool_loop(app, *a, **kw)
        app.tool_registry.execute.return_value = "Video generated at Media/out.mp4"
        app._run_blocking_inference.return_value = "Here is your neon city video."
        valid_resp = 'generate_video("neon city")'
        res2 = ChatbotApp._run_tool_loop(app, valid_resp, "make video", {})
        app.tool_registry.execute.assert_called_once_with("generate_video", {"prompt": "neon city"})
        self.assertEqual(res2, "Here is your neon city video.")

    def test_clean_latex_artifacts_stray_slash_n(self):
        """_clean_latex_artifacts converts stray literal '/n' tokens surrounded by whitespace into newlines."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        cleaned = ChatbotApp._clean_latex_artifacts(app, "Line one /n Line two")
        self.assertEqual(cleaned, "Line one \n Line two")
        # Path like C:/new/node should not be touched
        path_str = "Check path C:/new/node for details"
        self.assertEqual(ChatbotApp._clean_latex_artifacts(app, path_str), path_str)

    def test_load_secret_model_event_respects_auto_load_config(self):
        """_load_secret_model_event only calls model_swap if auto_load_on_unleash is True."""
        from main import ChatbotApp
        app = MagicMock(spec=ChatbotApp)
        app.depth_slider = MagicMock()
        app.config = {"auto_load_on_unleash": False}

        ChatbotApp._load_secret_model_event(app)
        self.assertEqual(app.active_persona_level, 7)
        app.model_swap.assert_not_called()

        app.config["auto_load_on_unleash"] = True
        ChatbotApp._load_secret_model_event(app)
        app.model_swap.assert_called_once_with(target_level=7)

    def test_extended_kv_cache_types_in_source(self):
        """Verify q6_0, turbo3_tcq, and turbo2_tcq are present in UNIVERSAL_KV in main.py."""
        with open("main.py", "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"q6_0"', src)
        self.assertIn('"turbo3_tcq"', src)
        self.assertIn('"turbo2_tcq"', src)

    def test_gemma4_unquoted_key_and_windows_path_tool_call(self):
        """Gemma-4 <|tool_call>call:read_file{path:"C:\\..."} with unquoted keys & raw backslashes parses and executes."""
        from System.tool_registry import GemmaToolRegistry
        from main import ChatbotApp

        app = MagicMock(spec=ChatbotApp)
        app.config = {}
        tr = GemmaToolRegistry(app)
        repro = (
            'User wants to read a specific file: `"C:\\Users\\ccrg6\\Desktop\\Hub\\Serenities\\SerenityPC\\TODO.txt"`.'
            'The available tool for reading files is `read_file(path: str)`.'
            'I will call `read_file` with the provided path.'
            '<|tool_call>call:read_file{path:"C:\\Users\\ccrg6\\Desktop\\Hub\\Serenities\\SerenityPC\\TODO.txt"}'
        )
        calls = tr.parse_tool_calls(repro)
        self.assertEqual(
            calls,
            [("read_file", {"path": r"C:\Users\ccrg6\Desktop\Hub\Serenities\SerenityPC\TODO.txt"})],
        )

        # Verify \t / \n in Windows paths and Gemma-4 <|"|> delimiters are preserved verbatim
        delim_call = '<|tool_call>call:read_file{path:<|"|>C:\\test\\notes.txt<|"|>}'
        self.assertEqual(
            tr.parse_tool_calls(delim_call),
            [("read_file", {"path": r"C:\test\notes.txt"})],
        )

        # End-to-end _run_tool_loop execution
        app.tool_registry = tr
        app.stop_process = MagicMock()
        app.stop_process.is_set.return_value = False
        app.process_queue = MagicMock()
        app.active_persona_level = 2
        app._run_tool_loop = lambda *a, **kw: ChatbotApp._run_tool_loop(app, *a, **kw)
        app._run_blocking_inference.return_value = "Here is TODO.txt content."
        with patch.object(tr, "execute", return_value="# SerenityPC TODO List") as mock_exec:
            res = ChatbotApp._run_tool_loop(app, repro, [{"role": "user", "content": "read TODO.txt"}], {})
        mock_exec.assert_called_once_with(
            "read_file", {"path": r"C:\Users\ccrg6\Desktop\Hub\Serenities\SerenityPC\TODO.txt"}
        )
        self.assertEqual(res, "Here is TODO.txt content.")

    def test_windows_drive_letter_not_split_as_kwarg_and_terminal_cwd(self):
        """Windows drive letters (C:\\) inside terminal commands/paths never fracture into {'C': '...'} and cwd executes cleanly."""
        import os
        from System.tool_registry import GemmaToolRegistry
        from main import ChatbotApp

        app = MagicMock(spec=ChatbotApp)
        app.config = {"expanded_file_access": True}
        tr = GemmaToolRegistry(app)

        target_dir = r"C:\Users\ccrg6\Desktop\Hub\Serenities\Environment M\Training Station\GGUForge"

        # 1. Gemma-4 <|"|> delimiters wrapping inner quotes + Windows drive path
        gemma_delim_cmd = (
            '<|tool_call>call:run_terminal_command'
            f'{{command:<|"|>dir "{target_dir}"<|"|>}}'
        )
        self.assertEqual(
            tr.parse_tool_calls(gemma_delim_cmd),
            [("run_terminal_command", {"command": f'dir "{target_dir}"'})],
        )

        # 2. Escaped inner quotes + raw single backslashes
        escaped_cmd = (
            '<|tool_call>call:run_terminal_command'
            f'{{command:"dir \\"{target_dir}\\""}}'
        )
        self.assertEqual(
            tr.parse_tool_calls(escaped_cmd),
            [("run_terminal_command", {"command": f'dir "{target_dir}"'})],
        )

        # 3. Unescaped inner quotes fallback (previously fractured into {'command': 'dir ', 'C': '\\Users...'})
        unescaped_cmd = (
            '<|tool_call>call:run_terminal_command'
            f'{{command:"dir "{target_dir}""}}'
        )
        parsed_unescaped = tr.parse_tool_calls(unescaped_cmd)
        self.assertEqual(len(parsed_unescaped), 1)
        self.assertNotIn("C", parsed_unescaped[0][1])
        self.assertEqual(parsed_unescaped[0], ("run_terminal_command", {"command": f'dir "{target_dir}"'}))

        # 4. Dedicated cwd parameter in tag call, JSON container, and Python PTC
        cwd_tag = f'<|tool_call>call:run_terminal_command{{command:"dir /s /b", cwd:"{target_dir}"}}'
        self.assertEqual(
            tr.parse_tool_calls(cwd_tag),
            [("run_terminal_command", {"command": "dir /s /b", "cwd": target_dir})],
        )

        cwd_json = f'<tool_call>{{"name": "run_terminal_command", "arguments": {{"command": "dir /b", "cwd": "{target_dir}"}}}}</tool_call>'
        self.assertEqual(
            tr.parse_tool_calls(cwd_json),
            [("run_terminal_command", {"command": "dir /b", "cwd": target_dir})],
        )

        cwd_ptc = f"run_terminal_command(command='dir /b', cwd='{target_dir}')"
        self.assertEqual(
            tr.parse_tool_calls(cwd_ptc),
            [("run_terminal_command", {"command": "dir /b", "cwd": target_dir})],
        )

        # 5. Schema exposes cwd and handler forwards target_cwd
        term_def = next(t["function"] for t in tr.get_definitions(level=2) if t["function"]["name"] == "run_terminal_command")
        self.assertIn("cwd", term_def["parameters"]["properties"])
        repo_cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        app._confirm_terminal_command = MagicMock(return_value=True)
        app.run_terminal_drawer_command = MagicMock(return_value="tool_registry.py")
        out = tr.execute("run_terminal_command", {"command": "dir /b", "cwd": repo_cwd})
        self.assertEqual(out, "tool_registry.py")
        app._confirm_terminal_command.assert_called_once_with("dir /b", repo_cwd)
        app.run_terminal_drawer_command.assert_called_once_with("dir /b", cwd=repo_cwd)

    def test_list_directory_header_and_ellipsis_path_recovery(self):
        """list_directory prepends Directory header and read_file resolves '.../file.json' safely against listed dirs."""
        import os
        import tempfile
        from System.tool_registry import GemmaToolRegistry

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tmp_parent = os.path.join(repo_root, ".tmp")
        os.makedirs(tmp_parent, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmp_dir:
            summary_p = os.path.join(tmp_dir, "moe_dual_refined_summary.json")
            with open(summary_p, "w", encoding="utf-8") as f:
                f.write('{"experts": 4, "status": "verified"}')
            env_p = os.path.join(tmp_dir, ".env")
            with open(env_p, "w", encoding="utf-8") as f:
                f.write("SECRET_TOKEN=blocked")

            app = MagicMock()
            app.script_dir = repo_root
            app.config = {"expanded_file_access": True}
            app.active_persona_level = 2
            tr = GemmaToolRegistry(app)

            ls_out = tr.execute("list_directory", {"path": tmp_dir})
            self.assertTrue(ls_out.startswith(f"Directory: {tmp_dir}\n"))
            self.assertIn("moe_dual_refined_summary.json", ls_out)
            self.assertEqual(tr._last_listed_dirs[0], os.path.realpath(tmp_dir))

            read_ok = tr.execute("read_file", {"path": ".../moe_dual_refined_summary.json"})
            self.assertIn('"experts": 4', read_ok)

            # Sensitive files must stay blocked even via ellipsis recovery
            read_env = tr.execute("read_file", {"path": ".../.env"})
            self.assertIn("SECURITY RESTRICTION", read_env)

            # Unresolvable ellipsis gives explicit full-path hint
            read_miss = tr.execute("read_file", {"path": ".../missing_xyz.json"})
            self.assertIn("Notice:", read_miss)
            self.assertIn("verbatim", read_miss)

    def test_run_tool_loop_multi_call_cache_full_query_and_override(self):
        """_run_tool_loop executes multiple calls, deduplicates via tool_cache, preserves full query, and emits T1 UI events."""
        from System.tool_registry import GemmaToolRegistry
        from main import ChatbotApp

        app = MagicMock(spec=ChatbotApp)
        app.config = {}
        app.active_persona_level = 2
        app.model_path = "gemma-4.gguf"
        app.stop_process = MagicMock()
        app.stop_process.is_set.return_value = False
        app.process_queue = MagicMock()
        tr = GemmaToolRegistry(app)
        app.tool_registry = tr
        app._run_tool_loop = lambda *a, **kw: ChatbotApp._run_tool_loop(app, *a, **kw)

        seen_prompts = []
        def _fake_infer(msgs, params):
            seen_prompts.append(msgs)
            if len(seen_prompts) == 1:
                # Try repeating identical cached call
                return '<|tool_call>call:read_file{path:"a.txt"}'
            return "Final grounded answer."

        app._run_blocking_inference = _fake_infer
        cache = {}
        evidence = []
        long_prompt = r'Query: C:\Users\demo\GGUForge deep dive. ' + ("context " * 400)
        first_resp = '<|tool_call>call:list_directory{path:"C:\\dir"}\n<|tool_call>call:read_file{path:"a.txt"}'

        with patch.object(tr, "execute", side_effect=["Directory: C:\\dir\n[FILE] a.txt", "Alpha contents"]) as mock_exec:
            res = ChatbotApp._run_tool_loop(
                app,
                first_resp,
                long_prompt,
                {},
                system_prompt_override="CUSTOM DEEP COOK SYS",
                tool_cache=cache,
                original_query=r'C:\Users\demo\GGUForge deep dive',
                collect_evidence=evidence,
                nested_cyc_tag="dc_cycle_1_0",
            )

        self.assertEqual(res, "Final grounded answer.")
        # Only 2 real executions (the repeated read_file("a.txt") hit tool_cache!)
        self.assertEqual(mock_exec.call_count, 2)
        self.assertEqual(len(evidence), 2)
        # system_prompt_override preserved (no [DIRECT STRIKE] hijack) and original_query kept at top
        self.assertEqual(seen_prompts[0][0]["content"], "CUSTOM DEEP COOK SYS")
        self.assertIn(r'C:\Users\demo\GGUForge deep dive', seen_prompts[0][1]["content"])
        self.assertNotIn("[DIRECT STRIKE]", seen_prompts[0][0]["content"])
        # T1 dropdown queue events emitted
        emitted_statuses = [c.args[0].get("status") for c in app.process_queue.put.call_args_list if c.args]
        self.assertIn("tool_call_start", emitted_statuses)
        self.assertIn("tool_call_done", emitted_statuses)


if __name__ == "__main__":
    unittest.main()
