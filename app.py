import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Accident Severity Detection System",
    page_icon="🚨",
    layout="wide"
)

# Class definitions
CLASS_NAMES = ['No Accident', 'Minor Accident', 'Moderate Accident', 'Severe Accident', 'Totalled Vehicle']
CLASS_WEIGHTS = [0.2, 0.5, 1.0, 1.3, 1.2]

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper function to fix state dict keys
def fix_state_dict(state_dict):
    """Remove 'model.' prefix from state dict keys if present"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Load models function
@st.cache_resource
def load_models(model_paths):
    models_dict = {}
    
    # VGG16
    if model_paths['VGG16'].exists():
        try:
            vgg16 = models.vgg16(pretrained=False)
            vgg16.classifier[6] = nn.Linear(4096, 5)
            state_dict = torch.load(model_paths['VGG16'], map_location=device)
            state_dict = fix_state_dict(state_dict)
            vgg16.load_state_dict(state_dict)
            vgg16.to(device)
            vgg16.eval()
            models_dict['VGG16'] = vgg16
            st.success("✅ VGG16 loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load VGG16: {str(e)}")
    
    # ResNet50
    if model_paths['ResNet50'].exists():
        try:
            resnet50 = models.resnet50(pretrained=False)
            # Use Sequential with Dropout as in training
            resnet50.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(2048, 5)
            )
            state_dict = torch.load(model_paths['ResNet50'], map_location=device)
            state_dict = fix_state_dict(state_dict)
            resnet50.load_state_dict(state_dict)
            resnet50.to(device)
            resnet50.eval()
            models_dict['ResNet50'] = resnet50
            st.success("✅ ResNet50 loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load ResNet50: {str(e)}")
    
    # DenseNet121
    if model_paths['DenseNet121'].exists():
        try:
            densenet121 = models.densenet121(pretrained=False)
            # Use Sequential with Dropout as in training
            densenet121.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1024, 5)
            )
            state_dict = torch.load(model_paths['DenseNet121'], map_location=device)
            state_dict = fix_state_dict(state_dict)
            densenet121.load_state_dict(state_dict)
            densenet121.to(device)
            densenet121.eval()
            models_dict['DenseNet121'] = densenet121
            st.success("✅ DenseNet121 loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load DenseNet121: {str(e)}")
    
    return models_dict

# Ensemble prediction
def ensemble_predict(models_dict, image_tensor, weights={'VGG16': 0.406, 'DenseNet121': 0.308, 'ResNet50': 0.286}):
    ensemble_probs = torch.zeros(5).to(device)
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            ensemble_probs += weights.get(model_name, 1.0) * probs
    
    ensemble_probs /= sum(weights.values())
    predicted_class = torch.argmax(ensemble_probs).item()
    confidence = ensemble_probs[predicted_class].item()
    
    return predicted_class, confidence, ensemble_probs.cpu().numpy()

# Send SMS via Twilio
def send_sms_twilio(to_phone, severity, confidence, timestamp, twilio_sid, twilio_token, twilio_from):
    try:
        from twilio.rest import Client
        
        client = Client(twilio_sid, twilio_token)
        
        message_body = f"""
🚨 EMERGENCY ALERT
Severity: {severity}
Confidence: {confidence:.2%}
Time: {timestamp}
Action required immediately!
"""
        
        message = client.messages.create(
            body=message_body,
            from_=twilio_from,
            to=to_phone
        )
        
        return True, message.sid
    except Exception as e:
        return False, str(e)

# Send email via SMTP2GO (free tier)
def send_email_smtp2go(to_email, severity, confidence, timestamp, smtp2go_api_key, from_email):
    try:
        import requests
        
        url = "https://api.smtp2go.com/v3/email/send"
        
        payload = {
            "api_key": smtp2go_api_key,
            "to": [to_email],
            "sender": from_email,
            "subject": f"🚨 EMERGENCY: {severity} Detected!",
            "text_body": f"""
EMERGENCY ALERT - VEHICLE ACCIDENT DETECTED

Severity: {severity}
Confidence: {confidence:.2%}
Timestamp: {timestamp}

Immediate action required. Emergency services should be contacted.

This is an automated message from the Accident Detection System.
""",
            "html_body": f"""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: #d32f2f;">🚨 EMERGENCY ALERT</h2>
<h3>Vehicle Accident Detected</h3>
<table style="border-collapse: collapse; margin: 20px 0;">
<tr><td style="padding: 8px; font-weight: bold;">Severity:</td><td style="padding: 8px; color: #d32f2f;">{severity}</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">Confidence:</td><td style="padding: 8px;">{confidence:.2%}</td></tr>
<tr><td style="padding: 8px; font-weight: bold;">Timestamp:</td><td style="padding: 8px;">{timestamp}</td></tr>
</table>
<p style="color: #d32f2f; font-weight: bold;">⚠️ Immediate action required. Emergency services should be contacted.</p>
<p style="color: #666; font-size: 12px;">This is an automated message from the Accident Detection System.</p>
</body>
</html>
"""
        }
        
        response = requests.post(url, json=payload)
        result = response.json()
        
        if result.get('data', {}).get('succeeded', 0) > 0:
            return True, "Email sent successfully"
        else:
            return False, result.get('data', {}).get('errors', 'Unknown error')
            
    except Exception as e:
        return False, str(e)

# Send webhook notification
def send_webhook(webhook_url, severity, confidence, timestamp):
    try:
        import requests
        
        payload = {
            "severity": severity,
            "confidence": f"{confidence:.2%}",
            "timestamp": timestamp,
            "alert_type": "vehicle_accident",
            "emergency": severity in ['Severe Accident', 'Totalled Vehicle']
        }
        
        response = requests.post(webhook_url, json=payload, timeout=5)
        
        if response.status_code == 200:
            return True, "Webhook triggered successfully"
        else:
            return False, f"Status code: {response.status_code}"
            
    except Exception as e:
        return False, str(e)

# Send email via free SMTP services (multiple fallbacks)
def send_email_free_smtp(to_email, severity, confidence, timestamp, from_email="accident.alert@system.com"):
    """Try multiple free SMTP services as fallback"""
    
    subject = f"🚨 EMERGENCY: {severity} Detected!"
    body = f"""
EMERGENCY ALERT - VEHICLE ACCIDENT DETECTED

Severity: {severity}
Confidence: {confidence:.2%}
Timestamp: {timestamp}

Immediate action required. Emergency services should be contacted.

This is an automated message from the Accident Detection System.
"""
    
    # List of free SMTP servers to try
    smtp_servers = [
        # SendGrid SMTP (requires free account)
        {'host': 'smtp.sendgrid.net', 'port': 587, 'name': 'SendGrid'},
        # Mailgun SMTP (requires free account)
        {'host': 'smtp.mailgun.org', 'port': 587, 'name': 'Mailgun'},
    ]
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    for smtp_config in smtp_servers:
        try:
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'], timeout=5)
            server.starttls()
            # Note: This still requires credentials, but keeping for reference
            server.quit()
        except:
            continue
    
    return False, "All SMTP servers failed"

# Play alarm sound
def play_alarm():
    audio_file = Path(r"working\emergency-alarm-69780.mp3")
    if audio_file.exists():
        audio_bytes = audio_file.read_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_tag = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        st.markdown(audio_tag, unsafe_allow_html=True)

# Trigger emergency response
def trigger_emergency_response(severity, confidence, emergency_contacts):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if emergency response needed
    if severity in ['Severe Accident', 'Totalled Vehicle']:
        st.error(f"🚨 EMERGENCY ALERT: {severity} detected with {confidence:.2%} confidence!")
        
        # Play alarm
        play_alarm()
        
        # Send notifications
        notification_cols = st.columns(3)
        
        # SMS via Twilio
        if emergency_contacts.get('twilio_enabled'):
            with notification_cols[0]:
                st.warning("📱 Sending SMS...")
                success, message = send_sms_twilio(
                    emergency_contacts['contact_phone'],
                    severity,
                    confidence,
                    timestamp,
                    emergency_contacts['twilio_sid'],
                    emergency_contacts['twilio_token'],
                    emergency_contacts['twilio_from']
                )
                if success:
                    st.success(f"✅ SMS sent! ID: {message}")
                else:
                    st.error(f"❌ SMS failed: {message}")
        
        # Email via SMTP2GO
        if emergency_contacts.get('smtp2go_enabled'):
            with notification_cols[1]:
                st.warning("📧 Sending Email...")
                success, message = send_email_smtp2go(
                    emergency_contacts['contact_email'],
                    severity,
                    confidence,
                    timestamp,
                    emergency_contacts['smtp2go_api_key'],
                    emergency_contacts.get('from_email', 'alert@accident-system.com')
                )
                if success:
                    st.success(f"✅ Email sent!")
                else:
                    st.error(f"❌ Email failed: {message}")
        
        # Webhook
        if emergency_contacts.get('webhook_enabled'):
            with notification_cols[2]:
                st.warning("🔗 Triggering Webhook...")
                success, message = send_webhook(
                    emergency_contacts['webhook_url'],
                    severity,
                    confidence,
                    timestamp
                )
                if success:
                    st.success(f"✅ Webhook triggered!")
                else:
                    st.error(f"❌ Webhook failed: {message}")
        
        # Emergency contact info display
        st.info("📞 Emergency Protocol Activated")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Severity:** {severity}")
            st.write(f"**Confidence:** {confidence:.2%}")
        with col2:
            st.write(f"**Time:** {timestamp}")
            if emergency_contacts.get('contact_phone'):
                st.write(f"**Contact:** {emergency_contacts['contact_phone']}")
    
    elif severity == 'Moderate Accident':
        st.warning(f"⚠️ WARNING: {severity} detected with {confidence:.2%} confidence")
        st.info("Medical attention may be required. Please assess the situation.")
    
    elif severity == 'Minor Accident':
        st.info(f"ℹ️ NOTICE: {severity} detected with {confidence:.2%} confidence")
        st.write("Minor damage detected. Vehicle inspection recommended.")
    
    else:
        st.success(f"✅ {severity} - No emergency response needed")

# Process single image
def process_image(image, models_dict, emergency_contacts):
    # Convert to PIL and transform
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 🔥 Force clean RGB (important fix)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = Image.fromarray(image)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    predicted_class, confidence, probs = ensemble_predict(models_dict, image_tensor)
    severity = CLASS_NAMES[predicted_class]
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Input Image", use_column_width=True)
    
    with col2:
        st.subheader("Classification Results")
        st.metric("Predicted Severity", severity)
        st.metric("Confidence", f"{confidence:.2%}")
        
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.progress(float(probs[i]), text=f"{class_name}: {probs[i]:.2%}")
    
    # Trigger emergency response
    trigger_emergency_response(severity, confidence, emergency_contacts)
    
    return severity, confidence

# Main application
def main():
    st.title("🚨 Vehicle Accident Severity Detection & Emergency Response System")
    st.markdown("---")
    
    # Sidebar for emergency contacts
    with st.sidebar:
        st.header("⚙️ Emergency Contact Configuration")
        
        # Select notification method
        notification_method = st.selectbox(
            "Select Notification Method",
            ["Twilio SMS (Recommended)", "SMTP2GO Email", "Webhook", "Multiple Methods"]
        )
        
        st.markdown("---")
        
        emergency_contacts = {}
        
        # Twilio SMS Configuration
        if notification_method in ["Twilio SMS (Recommended)", "Multiple Methods"]:
            st.subheader("📱 Twilio SMS Setup")
            st.info("Get FREE trial at [twilio.com/try-twilio](https://www.twilio.com/try-twilio)")
            
            twilio_sid = st.text_input("Twilio Account SID", type="password",
                                       help="From Twilio Console")
            twilio_token = st.text_input("Twilio Auth Token", type="password",
                                         help="From Twilio Console")
            twilio_from = st.text_input("Twilio Phone Number",
                                        help="Format: +1234567890")
            contact_phone = st.text_input("Emergency Contact Phone",
                                          help="Format: +1234567890")
            
            emergency_contacts.update({
                'twilio_enabled': bool(twilio_sid and twilio_token and twilio_from and contact_phone),
                'twilio_sid': twilio_sid,
                'twilio_token': twilio_token,
                'twilio_from': twilio_from,
                'contact_phone': contact_phone
            })
            
            if st.checkbox("Show Twilio Setup Instructions"):
                st.markdown("""
                **Quick Setup:**
                1. Go to [twilio.com/try-twilio](https://www.twilio.com/try-twilio)
                2. Sign up (FREE $15 credit)
                3. Get a phone number
                4. Copy Account SID & Auth Token
                5. Verify your emergency contact number
                """)
        
        # SMTP2GO Email Configuration
        if notification_method in ["SMTP2GO Email", "Multiple Methods"]:
            st.subheader("📧 SMTP2GO Email Setup")
            st.info("Get FREE account at [smtp2go.com](https://www.smtp2go.com/)")
            
            smtp2go_api_key = st.text_input("SMTP2GO API Key", type="password",
                                            help="From SMTP2GO Dashboard")
            from_email = st.text_input("From Email",
                                       value="alert@accident-system.com",
                                       help="Can be any email")
            contact_email = st.text_input("Emergency Contact Email")
            
            emergency_contacts.update({
                'smtp2go_enabled': bool(smtp2go_api_key and contact_email),
                'smtp2go_api_key': smtp2go_api_key,
                'from_email': from_email,
                'contact_email': contact_email
            })
            
            if st.checkbox("Show SMTP2GO Setup Instructions"):
                st.markdown("""
                **Quick Setup:**
                1. Go to [smtp2go.com](https://www.smtp2go.com/)
                2. Sign up (FREE 1000 emails/month)
                3. Go to Settings → API Keys
                4. Create new API key
                5. Copy and paste here
                """)
        
        # Webhook Configuration
        if notification_method in ["Webhook", "Multiple Methods"]:
            st.subheader("🔗 Webhook Setup")
            st.info("Use webhook.site for testing or your own endpoint")
            
            webhook_url = st.text_input("Webhook URL",
                                        help="POST request will be sent here")
            
            emergency_contacts.update({
                'webhook_enabled': bool(webhook_url),
                'webhook_url': webhook_url
            })
            
            if st.checkbox("Show Webhook Setup Instructions"):
                st.markdown("""
                **Quick Setup:**
                1. Go to [webhook.site](https://webhook.site/)
                2. Copy your unique URL
                3. Paste here
                4. View requests in real-time
                
                **Or use:**
                - IFTTT Webhooks
                - Zapier Webhooks
                - Discord/Slack Webhooks
                - Your own API endpoint
                """)
        
        st.markdown("---")
        st.subheader("📊 Model Information")
        st.info("Ensemble Model: VGG16 + DenseNet121 + ResNet50")
        st.write("**Severity Classes:**")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{i+1}. {class_name}")
    
    # Model paths
    model_paths = {
        'VGG16': Path("working/VGG16_best.pth"),
        'ResNet50': Path("working/ResNet50_best.pth"),
        'DenseNet121': Path("working/DenseNet121_best.pth")
    }
    
    # Load models
    with st.spinner("Loading models..."):
        models_dict = load_models(model_paths)
    
    if not models_dict:
        st.error("❌ No models loaded! Please check model paths.")
        return
    
    st.success(f"✅ Loaded {len(models_dict)} model(s): {', '.join(models_dict.keys())}")
    
    # Input mode selection
    st.header("📥 Select Input Mode")
    input_mode = st.radio("Choose input type:", 
                          ["📷 Upload Image", "🎬 Upload Video", "📹 Live Webcam"],
                          horizontal=True)
    
    st.markdown("---")
    
    # IMAGE INPUT
    if input_mode == "📷 Upload Image":
        st.subheader("Upload Image for Analysis")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            with st.spinner("Analyzing image..."):
                process_image(image, models_dict, emergency_contacts)
    
    # VIDEO INPUT
    elif input_mode == "🎬 Upload Video":
        st.subheader("Upload Video for Frame-by-Frame Analysis")
        uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # Save video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            
            # Process video
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.info(f"Video loaded: {total_frames} frames at {fps} FPS")
            
            frame_skip = st.slider("Analyze every Nth frame:", 1, 30, 10)
            
            if st.button("🎬 Start Video Analysis"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                result_placeholder = st.empty()
                
                frame_count = 0
                analyzed_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Analyze every Nth frame
                    if frame_count % frame_skip == 0:
                        analyzed_count += 1
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        with result_placeholder.container():
                            st.write(f"**Frame {frame_count}/{total_frames}**")
                            severity, confidence = process_image(frame_rgb, models_dict, emergency_contacts)
                            
                            # If severe accident detected, stop processing
                            if severity in ['Severe Accident', 'Totalled Vehicle']:
                                st.warning("🚨 Severe accident detected! Stopping analysis.")
                                break
                        
                        time.sleep(0.1)  # Small delay for display
                    
                    progress_bar.progress(frame_count / total_frames)
                    status_text.text(f"Processing: Frame {frame_count}/{total_frames}")
                
                cap.release()
                st.success(f"✅ Video analysis complete! Analyzed {analyzed_count} frames.")
    
    # LIVE WEBCAM (FIXED)
    elif input_mode == "📹 Live Webcam":
        st.subheader("Live Webcam Monitoring (Browser Camera)")
        st.warning("📸 Click below to capture a frame from your webcam")
    
        camera_image = st.camera_input("Take a photo")
    
        if camera_image is not None:
            image = Image.open(camera_image).convert('RGB')
    
            st.image(image, caption="Captured Frame", use_column_width=True)
    
            with st.spinner("Analyzing frame..."):
                severity, confidence = process_image(image, models_dict, emergency_contacts)
    
            if severity in ['Severe Accident', 'Totalled Vehicle']:
                st.error("🚨 CRITICAL: Severe accident detected!")

if __name__ == "__main__":
    main()
