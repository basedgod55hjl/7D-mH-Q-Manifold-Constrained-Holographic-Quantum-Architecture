import os
import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import requests
import threading
import json
import time

# LM Studio Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-vl-4b"

class AbrasaxUI:
    def __init__(self):
        self.messages = [{"role": "system", "content": "You are the ABRASAX GOD OS Agent, operating in a 7D Holographic Quantum Manifold."}]
        self.input_text = ""
        self.is_waiting_for_response = False
        
    def send_to_lm_studio(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        self.is_waiting_for_response = True
        
        def api_call():
            try:
                payload = {
                    "model": MODEL_NAME,
                    "messages": self.messages,
                    "temperature": 0.7,
                    "max_tokens": 512
                }
                response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    ai_reply = data['choices'][0]['message']['content']
                    self.messages.append({"role": "assistant", "content": ai_reply})
                else:
                    self.messages.append({"role": "assistant", "content": f"[SYSTEM ERROR] LM Studio returned {response.status_code}: {response.text}"})
            except Exception as e:
                self.messages.append({"role": "assistant", "content": f"[SYSTEM ERROR] Failed to connect to LM Studio at {LM_STUDIO_URL}. Error: {str(e)}"})
            finally:
                self.is_waiting_for_response = False

        threading.Thread(target=api_call, daemon=True).start()

    def render(self):
        # Configure ImGui Window
        imgui.set_next_window_size(600, 400, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_bg_alpha(0.85)  # Hovering semi-transparent effect
        
        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
        imgui.begin("ABRASAX GOD OS OVERLAY", flags=flags)
        
        imgui.text_colored("ABRASAX GOD OS - 7D QUANTUM MANIFOLD ACTIVE", 0.0, 1.0, 0.8)
        imgui.separator()
        
        # System Controls
        if imgui.button("Initialize Neural Lattice"):
            self.messages.append({"role": "system", "content": "[LATTICE INITIALIZED]"})
        imgui.same_line()
        if imgui.button("Deploy Crystal LLM"):
            self.messages.append({"role": "system", "content": "[CRYSTAL LLM DEPLOYED]"})
        imgui.same_line()
        if imgui.button("System Overwatch"):
            self.messages.append({"role": "system", "content": "[OVERWATCH ACTIVE]"})
            
        imgui.separator()
        
        # Chat display area
        imgui.begin_child("ChatRegion", width=0, height=-40, border=True)
        for msg in self.messages:
            if msg["role"] == "user":
                imgui.text_colored(f"USER: {msg['content']}", 0.5, 0.8, 1.0)
            elif msg["role"] == "assistant":
                imgui.text_wrapped(f"ABRASAX: {msg['content']}")
            elif msg["role"] == "system":
                imgui.text_colored(f"SYS: {msg['content']}", 0.4, 0.4, 0.4)
        
        if self.is_waiting_for_response:
            imgui.text_colored("ABRASAX is computing through the manifold...", 1.0, 1.0, 0.0)
            
        # Auto-scroll to bottom
        if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
            imgui.set_scroll_here_y(1.0)
            
        imgui.end_child()
        
        # Input area
        changed, self.input_text = imgui.input_text("##Input", self.input_text, 1024, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        imgui.same_line()
        if imgui.button("Transmit") or changed:
            if self.input_text.strip() and not self.is_waiting_for_response:
                self.send_to_lm_studio(self.input_text)
                self.input_text = ""
                
        imgui.end()

def main():
    if not glfw.init():
        print("Could not initialize OpenGL context")
        return

    # Set up a transparent, floating, frameless window
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    
    window = glfw.create_window(600, 400, "ABRASAX Overlay", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    # Bottom-right corner positioning
    monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(monitor)
    glfw.set_window_pos(window, video_mode.size.width - 620, video_mode.size.height - 440)

    imgui.create_context()
    impl = GlfwRenderer(window)
    
    ui = AbrasaxUI()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()
        
        # Apply dark futuristic theme
        style = imgui.get_style()
        style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.05, 0.05, 0.08, 0.85)
        style.window_rounding = 8.0
        style.frame_rounding = 4.0
        
        ui.render()
        
        imgui.render()
        
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        impl.render(imgui.get_draw_data())
        
        glfw.swap_buffers(window)
        
        # Omnipresence mode: no sleep for maximum responsiveness (set OMNIPRESENCE=1)
        if os.environ.get("OMNIPRESENCE") != "1":
            time.sleep(0.016)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
