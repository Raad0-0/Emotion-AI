# import gradio as gr
import gradio as gr
from scripts.main_logic import detect_audio_emotion, detect_text_emotion

# text = "I've fallen Asleeep a lot faster but I also feel like so funny"
# print(f"Text Emotion: {detect_text_emotion(text)}")

# audio =  'C:/Users/raada/Documents/Pray&Hope/data/predict_test/03-01-06-01-02-01-01.wav'
# print(f"Audio Emotion: {detect_audio_emotion(audio)}")

def predict(input_type, input_data):
    if input_type == "Text":
        return detect_text_emotion(input_data)
    else:
        return detect_audio_emotion(input_data)



with gr.Blocks() as demo:
    gr.Markdown("# Unified Emptoion Detection")
    with gr.Tabs():
        with gr.TabItem("Text"):
            text_input = gr.Textbox(label="Enter your text")
            text_button = gr.Button("Detect Emotion")
            text_output = gr.Textbox(label="Predicted Emotion")
            
            text_button.click(
                fn=lambda t: detect_text_emotion(t),
                inputs=text_input,
                outputs=text_output
            )

        with gr.TabItem("Audio"):
            audio_input = gr.Audio(type="filepath", label="Upload your audio file")
            audio_output = gr.Textbox(label="Predicted Emotion")
            audio_button = gr.Button("Detect Emotion")
            
            audio_button.click(
                fn=lambda a: detect_audio_emotion(a),
                inputs=audio_input,
                outputs=audio_output
            )
            
demo.launch()