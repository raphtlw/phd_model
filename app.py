import streamlit as st
import torch
import numpy as np
from model.wav2vec2 import Wav2Vec2
from decoder.ctc_decoder import decode_lattice
from phonetics.ipa import symbol_to_descriptor, to_symbol
import scipy.io.wavfile as wav
from scipy import signal
import io

# Page setup
st.set_page_config(page_title="Phoneme Recognition", page_icon="🎤")
st.title("🎤 Phoneme Recognition Demo")
st.caption("Record a sentence and see the IPA transcription.")

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        try:
            model = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

model = load_model()

if model:
    # Audio Input
    audio_value = st.audio_input("Record a voice note")

    if audio_value:
        # st.audio(audio_value, format='audio/wav')
        
        # Process audio
        try:
            # Read wav file
            # audio_value is a BytesIO-like object
            sample_rate, audio_data = wav.read(audio_value)
            
            # Convert to float
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                
            # Mix to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
                
            # Resample to 16000 if needed
            target_fs = 16000
            if sample_rate != target_fs:
                # Calculate number of samples
                num_samples = int(len(audio_data) * target_fs / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                
            # Standardize
            mean = audio_data.mean()
            std = audio_data.std()
            audio_data = (audio_data - mean) / (std + 1e-9)
            
            # Prepare tensor
            # Model expects (batch, time)
            input_tensor = torch.tensor(
                audio_data[np.newaxis, :],
                dtype=torch.float,
                device=device
            )
            
            # Run Inference automatically or on button? 
            # User asked "allows the user to record a sentence and then runs the model" -> implies separate step or sequence.
            # I'll add a 'Transcribe' button to be explicit.
            
            if st.button("Transcribe", type="primary"):
                with st.spinner("Recognizing..."):
                    with torch.no_grad():
                        y_pred, enc_features, cnn_features = model(input_tensor)
                        
                    # Decode
                    phone_sequence, enc_feats, cnn_feats, probs = decode_lattice(
                        lattice=y_pred[0].cpu().numpy(),
                        enc_feats=enc_features[0].cpu().numpy(),
                        cnn_feats=cnn_features[0].cpu().numpy(),
                    )
                    
                    # Convert to symbols
                    symbol_sequence = [to_symbol(i) for i in phone_sequence]
                    ipa_text = "".join(symbol_sequence)
                    
                    st.success("Recognition Complete!")
                    st.markdown(f"### Result: `{ipa_text}`")
                    
                    # Detailed breakdown
                    with st.expander("Detailed Analysis"):
                        st.write("Sequence of phones:")
                        st.write(symbol_sequence)
                        
                        st.write("Phone Descriptors:")
                        descriptor_data = [{"Symbol": s, "Descriptor": symbol_to_descriptor(s)} for s in symbol_sequence]
                        st.table(descriptor_data)

        except Exception as e:
            st.error(f"Error processing audio: {e}")
            st.info("Ensure you have scipy installed: `pip install scipy`")
