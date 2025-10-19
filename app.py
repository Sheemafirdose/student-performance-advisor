import os  # ⬅️ ADD THIS CRITICAL IMPORT
from flask import Flask, render_template, request, session, jsonify
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
import random
import re
from datetime import datetime
from flask import redirect

# ==================== STUDENT ADVISOR MODEL ====================
class StudentAdvisorModel:
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        self.templates = self._build_response_templates()
        self.patterns = self._build_nlp_patterns()
        
    def _build_knowledge_base(self):
        """Comprehensive educational knowledge base"""
        return {
            'study_techniques': {
                'pomodoro': "Pomodoro technique: 25min study, 5min break",
                'active_recall': "Active recall: Test yourself instead of re-reading",
                'spaced_repetition': "Spaced repetition: Review material at increasing intervals",
                'feynman': "Feynman technique: Teach concepts to someone else"
            },
            'subject_strategies': {
                'programming': "Practice coding daily, build projects, solve on LeetCode",
                'mathematics': "Understand concepts, practice problems, focus on weak areas",
                'theory_subjects': "Create notes, use mind maps, regular revisions",
                'practical_labs': "Prepare beforehand, document experiments, understand applications"
            },
            'career_paths': {
                'higher_studies': "Maintain high CGPA, research experience, strong recommendations",
                'placements': "Technical skills, projects, communication, internships",
                'entrepreneurship': "Problem-solving, networking, project experience",
                'research': "Publications, professor guidance, academic excellence"
            },
            'mental_health': {
                'stress_management': "Regular breaks, exercise, sleep, time management",
                'motivation': "Set small goals, track progress, reward achievements",
                'confidence': "Practice, preparation, positive self-talk, gradual challenges"
            }
        }
    
    def _build_response_templates(self):
        """GPT-like response templates for natural conversations"""
        return {
            'greeting': [
                "I've analyzed your academic profile, and here's my assessment:",
                "Based on your current performance, here are my recommendations:",
                "Let me provide you with personalized suggestions for improvement:"
            ],
            'strength_acknowledgment': [
                "Great job on {strength}! This shows your potential in {area}.",
                "I notice you're strong in {strength} - this is a valuable asset.",
                "Your {strength} is impressive and will help you in your academic journey."
            ],
            'improvement_focus': [
                "To reach the next level, focus on improving {area}.",
                "The main area needing attention is {area}. Here's how to improve:",
                "I recommend prioritizing {area} for significant performance gains."
            ],
            'action_plan': [
                "Here's a step-by-step plan to help you improve:",
                "Let me outline a clear action plan for you:",
                "Follow this structured approach for better results:"
            ],
            'encouragement': [
                "With consistent effort, you can definitely achieve {target}!",
                "Remember, small daily improvements lead to big results!",
                "You have the potential - it's about building the right habits!"
            ]
        }
    
    def _build_nlp_patterns(self):
        """NLP patterns for intelligent response generation"""
        return {
            'cgpa_patterns': {
                'excellent': (8.5, 10.0),
                'good': (7.0, 8.49),
                'average': (6.0, 6.99),
                'needs_improvement': (0.0, 5.99)
            },
            'attendance_patterns': {
                'excellent': (90, 100),
                'good': (80, 89),
                'concerning': (70, 79),
                'critical': (0, 69)
            },
            'study_hours_patterns': {
                'optimal': (25, 40),
                'adequate': (20, 24),
                'insufficient': (15, 19),
                'critical': (0, 14)
            }
        }
    
    def analyze_student_profile(self, student_data):
        """Deep NLP-based analysis"""
        analysis = {
            'performance_summary': '',
            'key_strengths': [],
            'critical_areas': [],
            'improvement_opportunities': [],
            'risk_factors': [],
            'potential_level': 'high'
        }
        
        # Intelligent CGPA Analysis
        cgpa_analysis = self._analyze_cgpa(student_data['total_cgpa'])
        analysis['performance_summary'] += cgpa_analysis['summary']
        analysis['key_strengths'].extend(cgpa_analysis['strengths'])
        analysis['critical_areas'].extend(cgpa_analysis['concerns'])
        
        # Attendance Analysis
        attendance_analysis = self._analyze_attendance(student_data['attendance'])
        analysis['performance_summary'] += " " + attendance_analysis['summary']
        analysis['critical_areas'].extend(attendance_analysis['concerns'])
        
        # Study Habits Analysis
        study_analysis = self._analyze_study_habits(student_data['study_hours'])
        analysis['performance_summary'] += " " + study_analysis['summary']
        analysis['improvement_opportunities'].extend(study_analysis['suggestions'])
        
        # Backlog Analysis
        if student_data['backlogs'] > 0:
            backlog_analysis = self._analyze_backlogs(student_data['backlogs'])
            analysis['critical_areas'].extend(backlog_analysis['concerns'])
            analysis['risk_factors'].append(backlog_analysis['risk'])
        
        # Extracurricular Analysis
        extracurricular_analysis = self._analyze_extracurricular(
            student_data['competitions'], 
            student_data['projects_internships']
        )
        analysis['improvement_opportunities'].extend(extracurricular_analysis['suggestions'])
        
        # Confidence Analysis
        confidence_analysis = self._analyze_confidence(student_data['confidence_level'])
        analysis['improvement_opportunities'].extend(confidence_analysis['suggestions'])
        
        return analysis
    
    def _analyze_cgpa(self, cgpa):
        """Intelligent CGPA analysis with contextual understanding"""
        if cgpa >= 8.5:
            return {
                'summary': f"Your CGPA of {cgpa}/10 is excellent and demonstrates strong academic capabilities.",
                'strengths': ["Outstanding academic performance", "Strong conceptual understanding"],
                'concerns': []
            }
        elif cgpa >= 7.0:
            return {
                'summary': f"With a CGPA of {cgpa}/10, you're performing well but have room for growth.",
                'strengths': ["Solid academic foundation", "Good learning ability"],
                'concerns': [f"Aim for 8.0+ CGPA to unlock better opportunities"]
            }
        elif cgpa >= 6.0:
            return {
                'summary': f"Your current CGPA of {cgpa}/10 indicates potential that needs better channeling.",
                'strengths': ["Basic understanding of subjects"],
                'concerns': ["Need significant academic improvement", "Focus on study techniques"]
            }
        else:
            return {
                'summary': f"A CGPA of {cgpa}/10 requires immediate attention and strategic improvement.",
                'strengths': [],
                'concerns': ["Critical academic performance", "Urgent intervention needed"]
            }
    
    def _analyze_attendance(self, attendance):
        """Contextual attendance analysis"""
        if attendance >= 90:
            return {
                'summary': "Your excellent attendance shows great discipline and commitment to learning.",
                'concerns': []
            }
        elif attendance >= 80:
            return {
                'summary': "Good attendance, but reaching 90%+ would maximize your learning potential.",
                'concerns': ["Slight improvement needed in regularity"]
            }
        elif attendance >= 70:
            return {
                'summary': "Your attendance needs attention as it might be affecting concept understanding.",
                'concerns': ["Moderate attendance concern", "Missing important classroom interactions"]
            }
        else:
            return {
                'summary': "Low attendance is significantly impacting your academic performance.",
                'concerns': ["Critical attendance issue", "Missing foundational concepts"]
            }
    
    def _analyze_study_habits(self, study_hours):
        """Study habits analysis with personalized suggestions"""
        if study_hours >= 25:
            return {
                'summary': "Your study commitment is excellent - focus now on optimizing techniques.",
                'suggestions': ["Try advanced study methods like active recall and spaced repetition"]
            }
        elif study_hours >= 20:
            return {
                'summary': "Good study routine, but increasing to 25+ hours with better techniques will help.",
                'suggestions': ["Implement Pomodoro technique", "Create structured study schedule"]
            }
        elif study_hours >= 15:
            return {
                'summary': "Your study hours are below optimal - this is likely affecting performance.",
                'suggestions': ["Increase to 20-25 hours weekly", "Focus on consistent daily schedule"]
            }
        else:
            return {
                'summary': "Insufficient study time is a major factor in academic challenges.",
                'suggestions': ["Immediately increase study hours", "Seek academic counseling"]
            }
    
    def _analyze_backlogs(self, backlogs):
        """Backlog analysis with risk assessment"""
        if backlogs == 1:
            return {
                'concerns': [f"You have {backlogs} backlog - address it this semester"],
                'risk': "Low risk with timely action"
            }
        elif backlogs <= 3:
            return {
                'concerns': [f"{backlogs} backlogs need strategic clearance plan"],
                'risk': "Medium risk - requires focused effort"
            }
        else:
            return {
                'concerns': [f"{backlogs} backlogs - this is critically affecting your academic progress"],
                'risk': "High risk - immediate intervention needed"
            }
    
    def _analyze_extracurricular(self, competitions, projects):
        """Extracurricular involvement analysis"""
        suggestions = []
        
        if competitions == 0:
            suggestions.append("Participate in coding competitions to enhance technical skills")
        if projects == 0:
            suggestions.append("Start building projects to gain practical experience")
        
        if competitions > 0 and projects > 0:
            suggestions.append("Great extracurricular involvement - continue building on this")
        
        return {'suggestions': suggestions}
    
    def _analyze_confidence(self, confidence_level):
        """Confidence level analysis"""
        if confidence_level >= 8:
            return {'suggestions': ["Maintain your high confidence - it's a great asset"]}
        elif confidence_level >= 6:
            return {'suggestions': ["Good confidence level - continue building through achievements"]}
        else:
            return {'suggestions': ["Work on confidence through small wins and preparation"]}
    
    def generate_advice(self, student_data, predicted_class):
        """Main method to generate GPT-like intelligent advice"""
        analysis = self.analyze_student_profile(student_data)
        
        # Build natural language response
        response_parts = []
        
        # Greeting
        response_parts.append(self._random_template('greeting'))
        
        # Performance summary
        response_parts.append(analysis['performance_summary'])
        
        # Strengths acknowledgment
        if analysis['key_strengths']:
            strength_text = self._random_template('strength_acknowledgment').format(
                strength=analysis['key_strengths'][0],
                area="academics" if "academic" in analysis['key_strengths'][0].lower() else "this area"
            )
            response_parts.append(strength_text)
        
        # Improvement focus
        if analysis['critical_areas']:
            focus_text = self._random_template('improvement_focus').format(
                area=analysis['critical_areas'][0].lower()
            )
            response_parts.append(focus_text)
        
        # Action plan
        response_parts.append(self._random_template('action_plan'))
        
        # Specific recommendations
        recommendations = self._generate_specific_recommendations(student_data, analysis)
        response_parts.extend(recommendations)
        
        # Encouragement
        target = self._get_target_performance(predicted_class)
        encouragement = self._random_template('encouragement').format(target=target)
        response_parts.append(encouragement)
        
        return "\n\n".join(response_parts)
    
    def _random_template(self, template_type):
        """Select random template for natural variation"""
        return random.choice(self.templates[template_type])
    
    def _generate_specific_recommendations(self, student_data, analysis):
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        # Academic recommendations
        if student_data['total_cgpa'] < 8.0:
            recommendations.append(
                f"🎯 **Academic Excellence Plan:**\n"
                f"• Target CGPA: 8.0+ (Current: {student_data['total_cgpa']}/10)\n"
                f"• Strategy: Identify 2 weakest subjects for focused improvement\n"
                f"• Action: Daily 1-hour dedicated study for each weak subject\n"
                f"• Resources: Faculty guidance + peer study groups"
            )
        
        # Attendance recommendations
        if student_data['attendance'] < 85:
            recommendations.append(
                f"📅 **Attendance Improvement:**\n"
                f"• Current: {student_data['attendance']}% → Target: 90%+\n"
                f"• Benefit: Better concept clarity + faculty rapport\n"
                f"• Tip: Set morning alarms + prepare notes night before\n"
                f"• Accountability: Study partner for mutual motivation"
            )
        
        # Study habits recommendations
        if student_data['study_hours'] < 20:
            recommendations.append(
                f"⏰ **Study Optimization:**\n"
                f"• Current: {student_data['study_hours']} hrs/week → Target: 25+ hrs\n"
                f"• Technique: Pomodoro (25min focus, 5min break)\n"
                f"• Schedule: 4-5 hours daily with variety in subjects\n"
                f"• Quality: Active learning over passive reading"
            )
        
        # Backlog recommendations
        if student_data['backlogs'] > 0:
            recommendations.append(
                f"🔧 **Backlog Clearance Strategy:**\n"
                f"• Current: {student_data['backlogs']} backlogs\n"
                f"• Priority: Clear easiest backlog first for momentum\n"
                f"• Schedule: 2 hours daily backlog study\n"
                f"• Goal: Clear 1-2 backlogs per semester"
            )
        
        # Skill development recommendations
        if student_data['competitions'] == 0 or student_data['projects_internships'] == 0:
            skill_text = "🚀 **Skill Development Roadmap:**\n"
            if student_data['competitions'] == 0:
                skill_text += "• Start with college-level coding competitions\n• Practice on HackerRank/LeetCode (30min daily)\n• Join programming clubs\n"
            if student_data['projects_internships'] == 0:
                skill_text += "• Build 2 mini-projects this semester\n• Learn Git and create GitHub portfolio\n• Apply for summer internships\n"
            recommendations.append(skill_text)
        
        return recommendations
    
    def _get_target_performance(self, current_class):
        """Get next performance level target"""
        levels = ['Below Average', 'Average', 'Good', 'Excellent']
        current_index = levels.index(current_class)
        if current_index < len(levels) - 1:
            return levels[current_index + 1]
        return "maintain your excellent performance"

# Initialize advisor model
advisor_model = StudentAdvisorModel()

# ==================== COMPREHENSIVE HELP SYSTEM ====================
class StudentHelpSystem:
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Comprehensive educational knowledge base"""
        return {
            'study_techniques': {
                'pomodoro': "🎯 **Pomodoro Technique**: Study for 25 minutes, then take a 5-minute break. After 4 cycles, take a longer 15-30 minute break. This improves focus and prevents burnout.",
                'active_recall': "🧠 **Active Recall**: Instead of re-reading, test yourself on the material. Use flashcards, practice questions, or teach the concepts to someone else.",
                'spaced_repetition': "📅 **Spaced Repetition**: Review material at increasing intervals (1 day, 3 days, 1 week, 2 weeks). Use apps like Anki or create a revision schedule.",
                'feynman': "💡 **Feynman Technique**: Choose a concept and explain it in simple terms as if teaching a child. Identify gaps in your understanding and simplify further."
            },
            
            'time_management': {
                'weekly_schedule': "⏰ **Weekly Schedule**: Create a timetable with fixed study slots. Include: 2-3 hours daily for core subjects, 1 hour for revisions, and regular breaks.",
                'priority_matrix': "🎯 **Eisenhower Matrix**: Categorize tasks as: 1. Urgent & Important (do now), 2. Important but not urgent (schedule), 3. Urgent but not important (delegate), 4. Neither (eliminate).",
                'productivity_tips': "🚀 **Productivity Tips**: Study during your peak energy hours, eliminate distractions (phone off), use the '2-minute rule' for small tasks, and track your progress weekly."
            },
            
            'subject_specific': {
                'programming': "💻 **Programming**: Practice daily on platforms like LeetCode/HackerRank. Build projects to apply concepts. Learn debugging techniques and version control with Git.",
                'mathematics': "📐 **Mathematics**: Understand concepts before solving problems. Practice regularly, focus on weak areas, and review previous years' question papers.",
                'theory_subjects': "📚 **Theory Subjects**: Create concise notes, use mind maps, teach concepts to others, and practice writing answers within time limits.",
                'practical_labs': "🔬 **Practical Labs**: Prepare beforehand, understand the theory behind experiments, document properly, and analyze results critically."
            },
            
            'exam_preparation': {
                'revision_strategy': "📖 **Revision Strategy**: 3-phase approach: 1. Quick overview (2 weeks before), 2. Detailed study (1 week before), 3. Final revision (last 3 days).",
                'time_management_exams': "⏱️ **Exam Time Management**: Divide time according to marks, attempt known questions first, keep last 15 minutes for review, and don't panic if stuck.",
                'stress_management': "😌 **Exam Stress Relief**: Practice deep breathing, get 7-8 hours sleep, eat healthy, take short breaks, and maintain positive self-talk."
            },
            
            'career_guidance': {
                'higher_studies': "🎓 **Higher Studies**: Maintain 8.0+ CGPA, gain research experience, build strong relationships with professors for recommendations, and prepare for entrance exams early.",
                'placements': "💼 **Placements**: Develop technical skills, build projects portfolio, practice communication skills, prepare for aptitude tests, and attend company presentations.",
                'internships': "🏢 **Internships**: Start applying 3-4 months in advance, tailor your resume for each role, prepare for interviews, and treat internships as learning opportunities.",
                'resume_building': "📄 **Resume Tips**: One-page format, action verbs, quantify achievements, include projects and skills, tailor for each application, and proofread carefully."
            },
            
            'mental_health': {
                'stress_management': "🌿 **Stress Management**: Regular exercise, 7-8 hours sleep, healthy diet, mindfulness meditation, and talking to friends/family.",
                'motivation': "🔥 **Staying Motivated**: Set small achievable goals, track progress, reward yourself, find study partners, and remember your long-term vision.",
                'burnout_prevention': "🛑 **Avoid Burnout**: Take regular breaks, maintain hobbies, set boundaries, get enough sleep, and don't compare yourself to others."
            },
            
            'campus_life': {
                'extracurricular': "🎭 **Extracurriculars**: Join clubs related to your interests, participate in college events, take leadership roles, and balance with academics.",
                'networking': "🤝 **Networking**: Attend workshops, connect with seniors and professors, participate in tech communities, and build your LinkedIn profile.",
                'time_balance': "⚖️ **Work-Life Balance**: Prioritize tasks, learn to say no, schedule fun activities, and maintain physical health alongside studies."
            }
        }
    
    def search_knowledge(self, query):
        """Search the knowledge base for relevant information"""
        query_lower = query.lower()
        results = []
        
        # Search through all categories
        for category, topics in self.knowledge_base.items():
            for topic, content in topics.items():
                if query_lower in topic.lower() or any(word in query_lower for word in topic.split()):
                    results.append({
                        'category': category.replace('_', ' ').title(),
                        'topic': topic.replace('_', ' ').title(),
                        'content': content
                    })
        
        # Also search in content for broader matches
        if not results:
            for category, topics in self.knowledge_base.items():
                for topic, content in topics.items():
                    if any(word in query_lower for word in content.lower().split()[:20]):  # Check first few words
                        results.append({
                            'category': category.replace('_', ' ').title(),
                            'topic': topic.replace('_', ' ').title(),
                            'content': content
                        })
        
        return results
    
    def get_help_categories(self):
        """Get all available help categories"""
        categories = {}
        for category, topics in self.knowledge_base.items():
            categories[category.replace('_', ' ').title()] = list(topics.keys())
        return categories

# Initialize help system
help_system = StudentHelpSystem()

# ==================== YOUR EXISTING FLASK APP ====================
app = Flask(__name__, template_folder='student_performance_dnn/templates')
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret_key_12345')
app.config['SESSION_PERMANENT'] = False
# KEEP ALL YOUR EXISTING CODE BELOW EXACTLY THE SAME

# Load models with error handling
try:
    scaler = joblib.load("student_performance_dnn/production_model/scaler.pkl")
    dnn_model = load_model("student_performance_dnn/production_model/student_performance_model.keras")
    label_encoder = joblib.load("student_performance_dnn/production_model/label_encoder.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # You might want to handle this more gracefully in production

# --- CORRECT Feature order that matches your NEW SCALER (8 features) ---
SCALER_FEATURES = [
    'total_cgpa', 
    'attendance', 
    'study_hours', 
    'backlogs', 
    'competitions', 
    'projects_internships', 
    'prevsem_cgpa',
    'confidence_level'
]

# Get class labels from your label encoder
CLASS_LABELS = list(label_encoder.classes_) if 'label_encoder' in locals() else ['Below Average', 'Average', 'Good', 'Excellent']

def fix_excellent_good_confusion(predicted_class, confidence, features_dict, probabilities):
    """
    Fix ONLY Excellent/Good classification issues
    """
    # Rule 1: If predicted as Excellent but CGPA < 8.0, likely should be Good
    if (predicted_class == 'Excellent' and 
        features_dict['total_cgpa'] < 8.0 and
        confidence < 0.85):  # Low confidence Excellent prediction
        
        if probabilities.get('Good', 0) > 0.15:  # Good probability is reasonable
            return 'Good'
    
    # Rule 2: If predicted as Good but has Excellent characteristics (CGPA > 8.5, no backlogs)
    elif (predicted_class == 'Good' and 
          features_dict['total_cgpa'] >= 8.5 and 
          features_dict['backlogs'] == 0 and
          features_dict['attendance'] >= 85 and
          confidence < 0.8):  # Low confidence Good prediction
        
        if probabilities.get('Excellent', 0) > 0.2:  # Excellent probability is reasonable
            return 'Excellent'
    
    return predicted_class

@app.route('/app', methods=['GET', 'POST'])
def main_app():   
    prediction_text = None
    error_text = None
    confidence_score = None
    probabilities = None

    if request.method == 'POST':
        try:
            # --- Collect ALL 8 features from form ---
            total_cgpa = float(request.form['total_cgpa'])
            prevsem_cgpa = float(request.form['prevsem_cgpa'])
            attendance = float(request.form['attendance'])
            
            # Validate ranges
            if not (0 <= total_cgpa <= 10):
                raise ValueError("Total CGPA must be between 0 and 10")
            if not (0 <= prevsem_cgpa <= 10):
                raise ValueError("Previous Semester CGPA must be between 0 and 10")
            if not (0 <= attendance <= 100):
                raise ValueError("Attendance must be between 0% and 100%")

            # Study hours mapping
            study_hours_map = {
                "0-10 (Minimal)": 5,
                "11-20 (Moderate)": 15,
                "21-30 (Regular)": 25,
                "31+ (Intensive)": 35
            }
            study_hours = study_hours_map[request.form['study_hours']]

            # Backlogs mapping
            backlogs_map = {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5+": 6
            }
            backlogs = backlogs_map[request.form['backlogs']]

            # Competitions / Projects - "More than 2" treated as "Yes"
            competitions_value = request.form['competitions']
            competitions = 1 if competitions_value in ["Yes", "More than 2"] else 0
            
            projects_value = request.form['projects_internships']
            projects_internships = 1 if projects_value in ["Yes", "More than 2"] else 0

            # Confidence level (NOW INCLUDED in scaling)
            confidence_level = int(request.form['confidence_level'])
            if not (1 <= confidence_level <= 10):
                raise ValueError("Confidence level must be between 1 and 10")

            # --- FIX: Prepare input with ALL 8 features for NEW scaler ---
            input_data = np.array([[
                total_cgpa,          # Feature 1
                attendance,          # Feature 2  
                study_hours,         # Feature 3
                backlogs,            # Feature 4
                competitions,        # Feature 5
                projects_internships, # Feature 6
                prevsem_cgpa,        # Feature 7
                confidence_level     # Feature 8 (NOW INCLUDED)
            ]])
            
            print(f"DEBUG: Input shape: {input_data.shape}")
            print(f"DEBUG: Features: {SCALER_FEATURES}")
            print(f"DEBUG: Input values: {input_data[0]}")
            
            # --- Scale ALL 8 features ---
            scaled_features = scaler.transform(input_data)
            print(f"DEBUG: Scaled shape: {scaled_features.shape}")
            print(f"DEBUG: Scaled values: {scaled_features[0]}")
            
            # --- Make prediction ---
            prediction_probs = dnn_model.predict(scaled_features)
            pred_class_idx = np.argmax(prediction_probs, axis=1)[0]
            predicted_class = label_encoder.inverse_transform([pred_class_idx])[0]
            confidence = np.max(prediction_probs)
            
            # Get all probabilities
            probabilities = {
                CLASS_LABELS[i]: float(prediction_probs[0][i]) * 100 
                for i in range(len(CLASS_LABELS))
            }
            
            # --- Apply Excellent/Good fix ---
            features_dict = {
                'total_cgpa': total_cgpa,
                'attendance': attendance,
                'backlogs': backlogs
            }
            
            final_prediction = fix_excellent_good_confusion(
                predicted_class, confidence, features_dict, 
                {k: v/100 for k, v in probabilities.items()}  # Convert back to 0-1 scale
            )
            
            confidence_score = confidence * 100
            prediction_text = f"Predicted Performance: {final_prediction}"

            # Store student data
            session['student_data'] = {
                'total_cgpa': total_cgpa,
                'attendance': attendance,
                'study_hours': study_hours,
                'backlogs': backlogs,
                'competitions': competitions,
                'projects_internships': projects_internships,
                'prevsem_cgpa': prevsem_cgpa,
                'confidence_level': confidence_level,
                'predicted_class': final_prediction
            }

        except Exception as e:
            error_text = f"❌ Error: {str(e)}"
            print(f"DEBUG Error: {e}")

    return render_template(
        'index.html', 
        prediction_text=prediction_text, 
        error_text=error_text,
        confidence_score=confidence_score,
        probabilities=probabilities
    )

@app.route('/')
def home():
    return render_template('home.html')  # Show landing page first

# ==================== SIMPLE SUGGESTION ROUTES ====================
# ==================== SIMPLE SUGGESTION ROUTES ====================
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    """Get personalized suggestions - FIXED FOR RENDER"""
    try:
        print("🔍 DEBUG: Attempting to get suggestions...")
        
        # Method 1: Try session first
        student_data = session.get('student_data')
        
        # Method 2: If session fails, use request data as backup
        if not student_data:
            request_data = request.get_json() or {}
            print(f"🔍 DEBUG: No session data, using request: {request_data}")
            
            if request_data:
                # Convert form data to proper format
                student_data = {
                    'total_cgpa': float(request_data.get('total_cgpa', 0)),
                    'attendance': float(request_data.get('attendance', 0)),
                    'study_hours': 5 if request_data.get('study_hours') == "0-10 (Minimal)" else 
                                  15 if request_data.get('study_hours') == "11-20 (Moderate)" else 
                                  25 if request_data.get('study_hours') == "21-30 (Regular)" else 
                                  35 if request_data.get('study_hours') == "31+ (Intensive)" else 0,
                    'backlogs': 0 if request_data.get('backlogs') == "0" else
                               1 if request_data.get('backlogs') == "1" else
                               2 if request_data.get('backlogs') == "2" else
                               3 if request_data.get('backlogs') == "3" else
                               4 if request_data.get('backlogs') == "4" else
                               6 if request_data.get('backlogs') == "5+" else 0,
                    'competitions': 1 if request_data.get('competitions') in ["Yes", "More than 2"] else 0,
                    'projects_internships': 1 if request_data.get('projects_internships') in ["Yes", "More than 2"] else 0,
                    'prevsem_cgpa': float(request_data.get('prevsem_cgpa', 0)),
                    'confidence_level': int(request_data.get('confidence_level', 5)),
                    'predicted_class': request_data.get('predicted_class', 'Average')
                }
                session['student_data'] = student_data
                print("✅ DEBUG: Created student data from request")
        
        if not student_data:
            return jsonify({
                'success': False,
                'message': 'Please fill out the form and click "Analyze Performance" first to get personalized suggestions.'
            })
        
        # Generate advice
        predicted_class = student_data.get('predicted_class', 'Average')
        advice = advisor_model.generate_advice(student_data, predicted_class)
        
        return jsonify({
            'success': True,
            'suggestions': advice
        })
        
    except Exception as e:
        print(f"❌ SUGGESTIONS ERROR: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error generating suggestions. Please try analyzing your performance first.'
        })

@app.route('/get_quick_suggestions', methods=['POST'])
def get_quick_suggestions():
    """Get quick suggestions for different categories"""
    try:
        category = request.json.get('category', 'study_tips')
        
        suggestions = {
            'study_tips': """
🎯 **Quick Study Tips:**
• Use Pomodoro technique (25min study, 5min break)
• Practice active recall instead of passive reading
• Create mind maps for complex topics
• Study during your peak energy hours
• Join study groups for difficult subjects
""",
            'exam_prep': """
📖 **Exam Preparation:**
• Start revision 2-3 weeks before exams
• Solve previous years' question papers
• Create concise notes for last-minute revision
• Practice time management during mock tests
• Get adequate sleep before exams
""",
            'time_management': """
⏰ **Time Management:**
• Create weekly study schedule
• Prioritize tasks using Eisenhower Matrix
• Set specific daily goals
• Eliminate distractions during study time
• Take regular breaks to avoid burnout
""",
            'career_advice': """
💼 **Career Guidance:**
• Build projects for your portfolio
• Practice coding on platforms like LeetCode
• Develop communication skills
• Network with seniors and professionals
• Prepare your resume with achievements
"""
        }
        
        return jsonify({
            'success': True,
            'suggestions': suggestions.get(category, "Select a valid category for suggestions.")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching quick suggestions: {str(e)}'
        })

# ==================== TOPIC SUGGESTIONS ROUTE ====================
@app.route('/get_topic_suggestions', methods=['POST'])
def get_topic_suggestions():
    """Get general topic suggestions"""
    try:
        topic = request.json.get('topic', '')
        
        suggestions = {
            'study_tips': """
🎯 **Study Techniques & Methods**

**Pomodoro Technique:**
• 25 minutes focused study + 5 minutes break
• After 4 cycles, take 15-30 minute longer break
• Improves concentration and prevents burnout

**Active Recall:**
• Test yourself instead of passive reading
• Use flashcards or practice questions
• Explain concepts without looking at notes

**Spaced Repetition:**
• Review at increasing intervals: 1 day, 3 days, 1 week, 2 weeks
• Use apps like Anki or create revision schedule
• Better long-term retention

**Feynman Technique:**
• Choose a concept and explain it in simple terms
• Identify gaps in your understanding
• Simplify and use analogies
""",
            'exam_prep': """
📖 **Exam Preparation Strategies**

**3-Phase Study Plan:**

**Phase 1: Foundation (4-6 weeks before)**
• Complete syllabus reading
• Create chapter-wise notes
• Identify important topics and weightage

**Phase 2: Intensive Practice (2-3 weeks before)**
• Solve previous years' question papers
• Take timed mock tests
• Focus on weak areas identified

**Phase 3: Revision (Last week)**
• Quick revision of notes
• Formula/theorem practice
• Time management practice

**Exam Day Tips:**
• Reach exam center early
• Read instructions carefully
• Attempt known questions first
• Keep last 15 minutes for review
""",
            'time_management': """
⏰ **Time Management & Productivity**

**Weekly Schedule Creation:**
• Fixed study slots for each subject
• 2-3 hours daily for core subjects
• 1 hour daily for revisions
• Include regular breaks

**Eisenhower Matrix:**
• **Do Now**: Urgent & Important tasks
• **Schedule**: Important but not urgent
• **Delegate**: Urgent but not important  
• **Eliminate**: Neither urgent nor important

**Productivity Tips:**
• Study during peak energy hours
• Eliminate distractions (phone off)
• Use the "2-minute rule" for small tasks
• Track progress weekly
""",
            'career_advice': """
💼 **Career Guidance & Placements**

**Placement Preparation Roadmap:**

**Technical Skills:**
• Data Structures & Algorithms
• Object-Oriented Programming
• DBMS, OS, Computer Networks
• Programming language proficiency

**Coding Practice:**
• LeetCode, HackerRank daily practice
• Focus on problem-solving patterns
• Time and space complexity analysis

**Projects & Portfolio:**
• 2-3 substantial projects on GitHub
• Good documentation and README files
• Live demos if possible

**Soft Skills:**
• Communication skills development
• Group discussion practice
• HR interview preparation
• Resume building with achievements
"""
        }
        
        return jsonify({
            'success': True,
            'suggestions': suggestions.get(topic, "Select a valid topic for suggestions.")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching topic suggestions: {str(e)}'
        })

# ==================== HELP SYSTEM ROUTES ====================
@app.route('/get_help_categories', methods=['GET'])
def get_help_categories():
    """Get all available help categories"""
    try:
        categories = help_system.get_help_categories()
        return jsonify({
            'success': True,
            'categories': categories
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching categories: {str(e)}'
        })

@app.route('/search_help', methods=['POST'])
def search_help():
    """Search the knowledge base for help topics"""
    try:
        query = request.json.get('query', '')
        if not query:
            return jsonify({
                'success': False,
                'message': 'Please provide a search query'
            })
        
        results = help_system.search_knowledge(query)
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error searching help: {str(e)}'
        })

# ==================== FALLBACK CHAT ROUTES ====================
@app.route('/start_chat', methods=['POST'])
def start_chat():
    """Fallback for chatbot - redirect to suggestions"""
    return jsonify({
        'response': "🎓 Welcome! I see you're interested in academic advice. Please use the 'GET PERSONALIZED ADVICE' button below for customized suggestions based on your performance data."
    })

@app.route('/send_message', methods=['POST'])
def send_message():
    """Fallback for chatbot messages"""
    return jsonify({
        'response': "I'm focusing on providing personalized academic suggestions through the main interface. Please fill out the form and use the 'GET PERSONALIZED ADVICE' button for detailed recommendations!"
    })

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    """Fallback for chat reset"""
    return jsonify({'status': 'success'})

# ==================== CLEAR SESSION ROUTE ====================
@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear session data"""
    session.clear()
    return jsonify({'status': 'success'})

# Diagnostic route to understand the issue
@app.route('/scaler_info')
def scaler_info():
    try:
        info = {
            'scaler_n_features': scaler.n_features_in_,
            'scaler_feature_names': getattr(scaler, 'feature_names_in_', 'Not available'),
            'model_input_shape': dnn_model.input_shape,
            'model_output_shape': dnn_model.output_shape,
            'class_labels': CLASS_LABELS
        }
        return f"""
        <h2>Model vs Scaler Info:</h2>
        <p><b>Scaler expects:</b> {info['scaler_n_features']} features</p>
        <p><b>Model expects:</b> {info['model_input_shape'][1]} features</p>
        <p><b>Class Labels:</b> {info['class_labels']}</p>
        <p><b>Features:</b> {SCALER_FEATURES}</p>
        <p><b>Status:</b> ✅ Scaler and Model both expect {info['scaler_n_features']} features</p>
        """
    except Exception as e:
        return f"Error: {e}"

# Health check route for Render
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
