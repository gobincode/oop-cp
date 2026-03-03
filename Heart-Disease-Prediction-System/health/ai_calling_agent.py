"""
AI Calling Agent for Automatic Appointment Booking
Uses Twilio Voice API with Gemini AI for intelligent conversations

AI Technology Used:
- Twilio Voice API (outbound calls)
- Google Gemini 2.5 Flash (Intelligent conversation AI)
- Twilio Speech Recognition (Speech-to-Text)
- Amazon Polly (Text-to-Speech with natural voices)

The AI can:
- Understand and respond to questions naturally
- Handle objections and concerns
- Provide patient information when asked
- Adapt conversation based on hospital staff responses
"""

import os
import urllib.request
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from django.conf import settings
import json
from datetime import datetime

CLAUDE_API_URL = 'https://api.quatarly.cloud/v1/messages'
CLAUDE_MODEL = 'claude-sonnet-4-6-thinking'

def call_claude(system_prompt: str, user_message: str, api_key: str, max_tokens: int = 200) -> str:
    """Call the Anthropic-compatible Claude API."""
    payload = json.dumps({
        'model': CLAUDE_MODEL,
        'max_tokens': max_tokens,
        'system': system_prompt,
        'messages': [{'role': 'user', 'content': user_message}]
    }).encode()
    req = urllib.request.Request(
        CLAUDE_API_URL,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return data['content'][0]['text'].strip()


class AICallingAgent:
    """
    AI Agent that calls hospitals to book cardiac appointments
    Uses Claude (Anthropic-compatible) for intelligent conversation handling
    """
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            self.client = None
        
        # Conversation context storage (in production, use Redis/database)
        self.conversation_history = {}
    
    def initiate_appointment_call(self, hospital_phone, patient_data, appointment_details):
        """
        Initiate an AI call to book an appointment (Twilio official pattern)
        
        Args:
            hospital_phone: Hospital/doctor phone number (format: +1234567890)
            patient_data: Dict with patient info (name, contact, age, etc.)
            appointment_details: Dict with appointment preferences
        
        Returns:
            call_sid: Twilio call SID for tracking
        """
        if not self.client:
            raise Exception("Twilio client not configured. Check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
        
        # Validate phone numbers are in E.164 format (+[country code][number])
        if not hospital_phone.startswith('+'):
            raise Exception(f"Phone number must be in E.164 format (e.g., +918087980346). Got: {hospital_phone}")
        
        if not self.phone_number.startswith('+'):
            raise Exception(f"Twilio phone number must be in E.164 format. Got: {self.phone_number}")
        
        # For testing, use test number
        test_number = os.getenv('TEST_PHONE_NUMBER')
        if test_number:
            hospital_phone = test_number  # Override with test number for development
        
        # Check if we have a live public URL for interactive AI conversation
        base_url = getattr(settings, 'BASE_URL', '') or ''
        base_url = base_url.rstrip('/')
        use_interactive_ai = bool(base_url) and any(
            kw in base_url for kw in ('ngrok', 'ngrok-free', 'herokuapp', 'render', 'appspot', 'railway', 'vercel')
        )
        
        try:
            if use_interactive_ai:
                # Use URL callbacks for interactive AI conversation
                print(f"🤖 Using interactive AI conversation with Gemini")
                print(f"   Callback URL: {base_url}/ai_call_handler/")
                
                # Encode patient data in URL
                import urllib.parse
                params = urllib.parse.urlencode({
                    'patient_name': patient_data.get('name', ''),
                    'patient_contact': patient_data.get('contact', ''),
                    'reason': appointment_details.get('reason', '')
                })
                
                call = self.client.calls.create(
                    url=f"{base_url}/ai_call_handler/?stage=greeting&{params}",
                    to=hospital_phone,
                    from_=self.phone_number,
                    status_callback=f"{base_url}/call_status/",
                    status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                    record=True,
                    machine_detection='DetectMessageEnd',
                )
                
                print(f"✅ Interactive AI call initiated! SID: {call.sid}")
            else:
                # Use simple TwiML message (no URL callbacks needed)
                print(f"📞 Using simple message (no interactive AI)")
                twiml = self._create_simple_twiml(patient_data, appointment_details)
                print(f"   Generated TwiML: {twiml[:200]}...")
                
                call = self.client.calls.create(
                    twiml=twiml,
                    to=hospital_phone,
                    from_=self.phone_number,
                    record=True,
                    machine_detection='DetectMessageEnd',
                )
                
                print(f"✅ Simple call initiated! SID: {call.sid}")
            
            print(f"   Status: {call.status}")
            print(f"   To: {call.to}")
            print(f"   From: {call._from}")
            
            # Persist call metadata in cache so call_status webhook can retrieve it
            from django.core.cache import cache
            cache.set(f'call_data_{call.sid}', {
                'patient_contact': patient_data.get('contact', ''),
                'patient_name': patient_data.get('name', ''),
                'hospital_name': appointment_details.get('hospital_name', ''),
                'hospital_phone': hospital_phone,
                'hospital_address': appointment_details.get('hospital_address', ''),
                'reason': appointment_details.get('reason', ''),
            }, timeout=7200)

            return call.sid
        
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Twilio API Error: {error_msg}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to initiate call: {error_msg}")
    
    def _create_simple_twiml(self, patient_data, appointment_details):
        """
        Create simple TwiML for local development (no URL callbacks needed)
        """
        response = VoiceResponse()
        
        # Complete message with all information
        patient_name = patient_data.get('name', 'a patient')
        patient_contact = patient_data.get('contact', 'not provided')
        reason = appointment_details.get('reason', 'cardiac consultation')
        
        message = (
            f"Hello, this is an automated appointment booking assistant calling on behalf of {patient_name}. "
            f"I'm calling to schedule a cardiac consultation appointment. "
            f"The patient recently had an ECG analysis showing concerning cardiac results. "
            f"The patient's contact number is {patient_contact}. "
            f"Please call back at your earliest convenience to schedule the appointment. "
            f"Thank you for your time and assistance. Goodbye."
        )
        
        response.say(message, voice='Polly.Joanna', language='en-US')
        response.hangup()
        
        return str(response)
    
    def _create_initial_twiml(self, patient_data, appointment_details, call_sid=None):
        """
        Create initial TwiML for the outbound call with AI conversation
        (Requires public URL - for production deployment)
        """
        response = VoiceResponse()
        
        # Initial greeting
        message = (
            f"Hello, this is an automated appointment booking assistant calling on behalf of "
            f"{patient_data.get('name', 'a patient')}. "
            f"I'm calling to schedule a cardiac consultation appointment. "
            f"Am I speaking with the appointment desk?"
        )
        
        response.say(message, voice='Polly.Joanna', language='en-US')
        
        # Start AI conversation
        gather = Gather(
            input='speech',
            action=f'/ai_call_handler/?stage=conversation&call_sid={call_sid}&patient_name={patient_data.get("name", "")}&patient_contact={patient_data.get("contact", "")}&reason={appointment_details.get("reason", "")}',
            method='POST',
            timeout=5,
            speech_timeout='auto',
            language='en-US'
        )
        response.append(gather)
        
        # If no response
        response.say("I didn't receive a response. I'll try again later. Goodbye.", voice='Polly.Joanna')
        response.hangup()
        
        return str(response)
    
    def generate_ai_script(self, patient_data, appointment_details):
        """
        Generate AI conversation script for booking appointment
        """
        script = f"""
        Hello, this is an automated appointment booking assistant calling on behalf of {patient_data['name']}.
        
        I'm calling to schedule a cardiac consultation appointment.
        
        Patient Details:
        - Name: {patient_data['name']}
        - Age: {patient_data.get('age', 'Not specified')}
        - Contact: {patient_data['contact']}
        - Reason: {appointment_details.get('reason', 'Cardiac consultation and ECG review')}
        
        Preferred appointment:
        - Date: {appointment_details.get('date', 'As soon as possible')}
        - Time: {appointment_details.get('time', 'Morning preferred')}
        
        Could you please help schedule this appointment?
        
        [Wait for response and confirm details]
        
        Thank you for your assistance. The patient will receive a confirmation message.
        """
        
        return script
    
    def _get_history(self, call_sid):
        from django.core.cache import cache
        return cache.get(f'call_history_{call_sid}', [])

    def _save_history(self, call_sid, history):
        from django.core.cache import cache
        cache.set(f'call_history_{call_sid}', history, timeout=3600)

    def _extract_and_cache_booking(self, call_sid, history):
        """Ask Claude to extract structured appointment details from conversation and cache them."""
        from django.core.cache import cache
        import json as _json
        conversation_text = '\n'.join(history[-12:])
        extract_system = (
            'Extract appointment details from this call transcript. '
            'Return ONLY valid JSON with keys: date_str, time_str, doctor_name, department. '
            'For date_str use plain English like "tomorrow" or "March 5". '
            'Leave a key empty string if not mentioned.'
        )
        try:
            raw = call_claude(extract_system, conversation_text, self.claude_api_key, max_tokens=120)
            raw = raw.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
            details = _json.loads(raw)
        except Exception as e:
            print(f'Booking detail extraction failed: {e}')
            details = {}
        cache.set(f'call_booking_{call_sid}', details, timeout=7200)
        print(f'Cached booking details for {call_sid}: {details}')

    def generate_ai_response(self, call_sid, user_speech, patient_data, conversation_stage):
        """
        Use Claude AI to generate intelligent responses based on conversation context.
        Returns (response_text, is_done) tuple.
        """
        if not self.claude_api_key:
            return self._fallback_response(conversation_stage), False

        history = self._get_history(call_sid)

        system_prompt = f"""You are a concise medical appointment booking assistant calling a hospital on behalf of a patient.

Patient: {patient_data.get('name', 'the patient')}
Contact: {patient_data.get('contact', 'not provided')}
Reason: Cardiac consultation - recent ECG showed concerning results

Rules:
- Maximum 2 sentences per reply. Be direct and efficient.
- Ask only ONE question at a time.
- Do NOT repeat information already confirmed.
- When the appointment date and time are confirmed, say a brief thank you and goodbye, then end your reply with [BOOKING_COMPLETE].
- If they ask for the patient's contact, provide it.
- Never re-introduce yourself after the greeting."""

        history.append(f"Staff: {user_speech}")
        user_message = "Conversation history:\n" + "\n".join(history[-8:]) + "\n\nYour reply:"

        try:
            ai_response = call_claude(system_prompt, user_message, self.claude_api_key, max_tokens=120)
            ai_response = ai_response.replace('"', '').replace('*', '').strip()

            is_done = '[BOOKING_COMPLETE]' in ai_response
            ai_response = ai_response.replace('[BOOKING_COMPLETE]', '').strip()

            if len(ai_response) > 280:
                ai_response = ai_response[:277] + '...'

            history.append(f"Assistant: {ai_response}")
            self._save_history(call_sid, history)

            if is_done:
                self._extract_and_cache_booking(call_sid, history)

            return ai_response, is_done
        except Exception as e:
            print(f"Claude AI error: {str(e)}")
            return self._fallback_response(conversation_stage), False
    
    def _fallback_response(self, stage):
        """Fallback responses if Claude is unavailable"""
        fallbacks = {
            'greeting': "Hello, I'm calling to book a cardiac consultation appointment. Is this the appointment desk?",
            'provide_details': "I need to book an appointment for a patient with concerning ECG results. What dates are available?",
            'confirm': "Thank you. The patient will call to confirm. Have a great day!",
        }
        return fallbacks.get(stage, "Thank you for your time. Goodbye."), False
    
    def create_twiml_response(self, stage='greeting', data=None, call_sid=None):
        """
        Create TwiML response with Gemini AI integration
        """
        response = VoiceResponse()
        
        if stage == 'greeting':
            # Initial greeting with patient info
            patient_data = data.get('patient_data', {}) if data else {}
            patient_name = patient_data.get('name', 'a patient')
            
            message = (
                f"Hello, this is an automated appointment booking assistant calling on behalf of {patient_name}. "
                f"I'm calling to schedule a cardiac consultation appointment. "
                f"Am I speaking with the appointment desk?"
            )
            
            response.say(message, voice='Polly.Joanna', language='en-US')
            
            # Encode patient data for next stage
            import urllib.parse
            params = urllib.parse.urlencode({
                'patient_name': patient_data.get('name', ''),
                'patient_contact': patient_data.get('contact', ''),
                'reason': patient_data.get('reason', '')
            })
            
            # Gather response with AI processing
            gather = Gather(
                input='speech',
                action=f'/ai_call_handler/?stage=conversation&call_sid={call_sid}&{params}',
                method='POST',
                timeout=5,
                speech_timeout='auto',
                language='en-US'
            )
            response.append(gather)
            
            # If no response
            response.say("I didn't receive a response. I'll try again later. Goodbye.")
            response.hangup()
        
        elif stage == 'conversation':
            import urllib.parse
            user_speech = data.get('SpeechResult', '')
            patient_data = data.get('patient_data', {})

            # Build params to carry patient context through every turn
            params = urllib.parse.urlencode({
                'patient_name': patient_data.get('name', ''),
                'patient_contact': patient_data.get('contact', ''),
                'reason': patient_data.get('reason', ''),
            })
            next_action = f'/ai_call_handler/?stage=conversation&call_sid={call_sid}&{params}'

            if user_speech:
                ai_message, is_done = self.generate_ai_response(
                    call_sid, user_speech, patient_data, 'conversation'
                )
                response.say(ai_message, voice='Polly.Joanna', language='en-US')

                if is_done:
                    response.hangup()
                else:
                    gather = Gather(
                        input='speech',
                        action=next_action,
                        method='POST',
                        timeout=5,
                        speech_timeout='auto',
                        language='en-US'
                    )
                    response.append(gather)
                    response.say("I didn't catch that. Goodbye.", voice='Polly.Joanna')
                    response.hangup()
            else:
                response.say("I didn't catch that. Could you please repeat?", voice='Polly.Joanna')
                gather = Gather(
                    input='speech',
                    action=next_action,
                    method='POST',
                    timeout=5,
                    speech_timeout='auto',
                    language='en-US'
                )
                response.append(gather)
                response.hangup()
        
        elif stage == 'confirm_appointment':
            # Final confirmation
            response.say(
                "Perfect. I'll confirm those details with the patient. "
                "They will call back to verify if needed. "
                "Thank you for your assistance. Have a great day!",
                voice='Polly.Joanna',
                language='en-US'
            )
            response.hangup()
        
        return str(response)
    
    def send_sms_confirmation(self, patient_phone, appointment_details):
        """
        Send SMS confirmation to patient after booking
        """
        if not self.client:
            print("❌ Twilio client not configured")
            return False
        
        # Format phone number
        if not patient_phone.startswith('+'):
            patient_phone = f"+{patient_phone}"
        
        message_body = f"""🏥 Appointment Booking Confirmation

Hospital: {appointment_details.get('hospital_name', 'N/A')}
Address: {appointment_details.get('hospital_address', 'N/A')}
Phone: {appointment_details.get('hospital_phone', 'N/A')}

Date: {appointment_details.get('date', 'To be confirmed')}
Time: {appointment_details.get('time', 'To be confirmed')}

Status: {appointment_details.get('status', 'Pending confirmation')}

Please call the hospital to confirm your appointment.

- Heart Disease Prediction System"""
        
        try:
            # Send SMS
            sms = self.client.messages.create(
                body=message_body,
                from_=self.phone_number,
                to=patient_phone
            )
            print(f"✅ SMS sent successfully! SID: {sms.sid}")
            return sms.sid
        except Exception as e:
            print(f"❌ Failed to send SMS: {str(e)}")
            return False
    
    def send_whatsapp_confirmation(self, patient_phone, appointment_details):
        """
        Send WhatsApp confirmation to patient after booking (Plain Text)
        """
        if not self.client:
            print("❌ Twilio client not configured")
            return False
        
        # Format phone number for WhatsApp
        if not patient_phone.startswith('+'):
            patient_phone = f"+{patient_phone}"
        
        # WhatsApp requires 'whatsapp:' prefix
        whatsapp_to = f"whatsapp:{patient_phone}"
        
        # Use WhatsApp sandbox number from environment or default
        whatsapp_sandbox = os.getenv('WHATSAPP_SANDBOX_NUMBER', '+14155238886')
        whatsapp_from = f"whatsapp:{whatsapp_sandbox}"
        
        # Create plain text message with appointment details
        message_body = f"""🏥 *Appointment Booking Confirmation*

*Hospital:* {appointment_details.get('hospital_name', 'N/A')}
*Address:* {appointment_details.get('hospital_address', 'N/A')}
*Phone:* {appointment_details.get('hospital_phone', 'N/A')}

*Date:* {appointment_details.get('date', 'To be confirmed')}
*Time:* {appointment_details.get('time', 'To be confirmed')}

*Status:* {appointment_details.get('status', 'Pending confirmation')}

Please call the hospital to confirm your appointment.

_Heart Disease Prediction System_"""
        
        try:
            # Send WhatsApp message with plain text
            whatsapp = self.client.messages.create(
                from_=whatsapp_from,
                body=message_body,
                to=whatsapp_to
            )
            print(f"✅ WhatsApp sent successfully! SID: {whatsapp.sid}")
            print(f"   Mode: Plain text message")
            return whatsapp.sid
        except Exception as e:
            print(f"❌ Failed to send WhatsApp: {str(e)}")
            print(f"   Note: Make sure you've joined the WhatsApp sandbox")
            print(f"   Send 'join <code>' to {whatsapp_sandbox} on WhatsApp")
            return False
    
    def send_appointment_notifications(self, patient_phone, appointment_details):
        """
        Send both SMS and WhatsApp notifications
        """
        results = {
            'sms': False,
            'whatsapp': False
        }
        
        print(f"\n📱 Sending appointment notifications to {patient_phone}")
        print(f"   Hospital: {appointment_details.get('hospital_name', 'N/A')}")
        
        # Send SMS
        sms_sid = self.send_sms_confirmation(patient_phone, appointment_details)
        results['sms'] = bool(sms_sid)
        
        # Send WhatsApp
        whatsapp_sid = self.send_whatsapp_confirmation(patient_phone, appointment_details)
        results['whatsapp'] = bool(whatsapp_sid)
        
        return results


def create_simple_booking_call(hospital_phone, patient_name, patient_contact, reason, hospital_name=None, hospital_address=None):
    """
    Simplified function to create an appointment booking call with hospital details
    """
    agent = AICallingAgent()
    
    patient_data = {
        'name': patient_name,
        'contact': patient_contact,
    }
    
    appointment_details = {
        'reason': reason,
        'date': 'As soon as possible',
        'time': 'Morning preferred',
        'hospital_name': hospital_name or 'Hospital',
        'hospital_phone': hospital_phone,
        'hospital_address': hospital_address or 'Address not provided',
    }
    
    try:
        call_sid = agent.initiate_appointment_call(
            hospital_phone,
            patient_data,
            appointment_details
        )
        return {
            'success': True,
            'call_sid': call_sid,
            'message': 'AI agent is calling the hospital to book your appointment'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to initiate AI call. Please try calling manually.'
        }
