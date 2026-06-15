import streamlit as st
import tempfile
import os
from pathlib import Path
from modules.modal_transcriber import ModalTranscriber
from modules.transcriber import ModelType, TargetLanguage
from modules.translation import LLMTranslator
from modules.extractor import LLMExtractor
from modules.comparator import LLMComparator

# Module initialization
@st.cache_resource
def load_modules():
    return {
        "transcriber": ModalTranscriber(),
        "translator": LLMTranslator(),
        "extractor": LLMExtractor(),
        "comparator": LLMComparator()
    }

modules = load_modules()

# UI display configuration
st.set_page_config(page_title="Wobsongo AI Pipeline", layout="wide")
st.title("Wobsongo Audio AI Pipeline")

# User input area
with st.container():
    st.subheader("1. Data Input")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        uploaded_audio = st.file_uploader("Upload File Audio")
        st.markdown("**OR**")
        input_audio_url = st.text_input("Use Audio URLs:")
        
    with col2:
        model_choice = st.selectbox(
            "ASR Model:",
            options=[
                "Omnilingual ASR",
                "Whisper Small (Original)",
                "Whisper Large-V3 (Original)",
                "Whisper Large-V3 (With Augmentation)",
                "Whisper Small (With Augmentation)",
                "Whisper Large-V3 (No Augmentation)",
                "Whisper Small (No Augmentation)"

            ],
            help="Select the AI model used to transcribe the audio."
        )

        target_language_choice = st.selectbox(
            "Target Translation:", 
            options=["English", "French"]
        )

        source_language_choice = st.selectbox(
            "Source Language (Audio):",
            options=["auto", "french", "english", "moore", "dioula"],
            help="Select a specific language for the audio to be transcribed."
        )
        
    with col3:
        human_transcript = st.text_area(
            "Human Transcription (Optional):", 
            height=150, 
            placeholder="Enter manual transcription here to compare with machine results."
        )

# Execution button
if st.button("Start the Analysis Process", type="primary", use_container_width=True):
    if not uploaded_audio and not input_audio_url:
        st.warning("Please upload the audio file OR provide an audio URL first!")
    else:
        # Specify the target language for ASR based on the dropdown selection.
        if target_language_choice == "English":
            asr_target_lang = TargetLanguage.ENGLISH
        else:
            asr_target_lang = TargetLanguage.FRENCH

        try:
            st.divider()

            audio_path = None
            if uploaded_audio:
                file_extension = Path(uploaded_audio.name).suffix 
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_audio.getvalue())
                    audio_path = Path(tmp_file.name)
            
            # Transcription (OmniASR via Modal)
            st.subheader("ASR Transcription")
            with st.spinner("The ASR engine is working..."):
                asr_result = modules["transcriber"].transcribe(
                    model=model_choice,
                    target_lang=asr_target_lang,
                    source_lang=source_language_choice,
                    audio=audio_path,
                    audio_url=input_audio_url.strip() if input_audio_url else None
                )
                st.success(f"Transcription Completed! Language detected: **{asr_result.language_selected}**")
                
                with st.expander("View Raw Transcript"):
                    st.write(asr_result.transcript)

            # Translation (OpenAI)
            st.subheader(f"Translation into {target_language_choice}")
            with st.spinner(f"Translating text into {target_language_choice}..."):
                # Translate ASR Machine Results
                machine_translation = modules["translator"].translate_text(
                    text=asr_result.transcript, 
                    source_lang=asr_result.language_selected,
                    target_lang=target_language_choice
                )
                
                # Translate human transcript
                human_translate = None
                if human_transcript.strip():
                    human_translate = modules["translator"].translate_text(
                        text=human_transcript, 
                        source_lang=asr_result.language_selected,
                        target_lang=target_language_choice
                    )

                # Show Translation Results
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**Machine Translation (ASR):**")
                    st.info(machine_translation.translated_text)
                
                with col_t2:
                    if human_translate:
                        st.markdown("**Human Translation (Manual):**")
                        st.success(human_translate.translated_text)
                    else:
                        st.markdown("**Human Translation (Manual):**")
                        st.warning("No human transcript data is input.")

            # Intent and Summary Extraction
            st.subheader("Intent & Data Extraction")
            with st.spinner("Analyzing the meaning and topic..."):
                # Extract from Machine Translation
                extract_machine = modules["extractor"].extract_data(machine_translation.translated_text)
                
                # Extract from Human Transcript
                human_extract = None
                if human_translate:
                    human_extract = modules["extractor"].extract_data(human_translate.translated_text)

                # Showing Machine and Human Extraction side by side
                col_e1, col_e2 = st.columns(2)
                
                with col_e1:
                    st.markdown("#### Analysis Results (Machine)")
                    st.write(f"**Summary:** {extract_machine.summary}")
                    st.write(f"**Topics:** {', '.join(extract_machine.topics)}")
                    with st.expander("View Claim Details & Raw JSON (Machine)"):
                        st.json(extract_machine.raw_json_string)
                
                with col_e2:
                    if human_extract:
                        st.markdown("#### Analysis Results (Human)")
                        st.write(f"**Summary:** {human_extract.summary}")
                        st.write(f"**Topics:** {', '.join(human_extract.topics)}")
                        with st.expander("View Claim Details & Raw JSON (Human)"):
                            st.json(human_extract.raw_json_string)
                    else:
                        st.markdown("#### Analysis Results (Human)")
                        st.warning("No human transcript data is input.")

            # Comparison (If human transcript available)
            if human_extract:
                st.divider()
                st.subheader("Semantic Similarity Evaluation")
                with st.spinner("Calculating Similarity..."):
                    
                    # Calculate Similarity Value
                    raw_sim = modules["comparator"].calculate_similarity(
                        human_translate.translated_text, 
                        machine_translation.translated_text
                    )
                    
                    sum_sim = modules["comparator"].calculate_similarity(
                        human_extract.summary, 
                        extract_machine.summary
                    )
                    
                    topic_h_str = ", ".join(human_extract.topics)
                    topic_m_str = ", ".join(extract_machine.topics)
                    topic_sim = modules["comparator"].calculate_similarity(topic_h_str, topic_m_str)
                    
                    st.markdown("Percentage of semantic similarity between human and machine transcript:")
                    m1, m2, m3 = st.columns(3)
                    
                    m1.metric(
                        label="Raw Text Similarity", 
                        value=f"{raw_sim}%",
                        help="Compares the full, raw text. A high score means the ASR captured almost all words and details exactly as the human did."
                    )
                    
                    m2.metric(
                        label="Summary Similarity", 
                        value=f"{sum_sim}%",
                        help="Compares only the AI generated summaries. A high score means the core message/story was successfully preserved, even if the ASR misheard some specific words."
                    )
                    
                    m3.metric(
                        label="Topic/Intent Similarity", 
                        value=f"{topic_sim}%",
                        help="Compares only the core topics and intents. A high score proves the ASR correctly identified WHAT the conversation is about, regardless of sentence structure."
                    )
        
        finally:
            if audio_path and audio_path.exists():
                os.remove(audio_path)