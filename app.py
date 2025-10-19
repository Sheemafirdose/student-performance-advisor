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

# ==================== CHAT ADVISOR ====================
class ChatAdvisor:
    def __init__(self):
        self.conversations = {}
    
    def handle_message(self, session_id, user_message, student_data=None):
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'step': 'greeting',
                'name': None
            }
        
        conv = self.conversations[session_id]
        user_msg = user_message.strip()
        
        if conv['step'] == 'greeting':
            conv['step'] = 'get_name'
            return "Hello! I'm your academic advisor. What's your name?"
        
        elif conv['step'] == 'get_name':
            if len(user_msg) < 2:
                return "Please enter a valid name:"
            conv['name'] = user_msg
            conv['step'] = 'show_suggestions'
            return f"Nice to meet you, {conv['name']}! I can analyze your academic data and provide personalized suggestions. Would you like me to do that? (yes/no)"
        
        elif conv['step'] == 'show_suggestions':
            user_lower = user_msg.lower()
            if any(word in user_lower for word in ['yes', 'yeah', 'sure', 'ok', 'yep']):
                if student_data:
                    # Generate advice using the main advisor model
                    advice = advisor_model.generate_advice(student_data, student_data.get('predicted_class', 'Average'))
                    conv['step'] = 'completed'
                    return f"Great! Here are my personalized suggestions for you, {conv['name']}:\n\n{advice}"
                else:
                    return "I don't have your academic data. Please submit the form first."
            elif any(word in user_lower for word in ['no', 'not', 'nope', 'later']):
                conv['step'] = 'completed'
                return f"No problem {conv['name']}! Feel free to ask anytime you need academic advice."
            else:
                return "Please answer with 'yes' or 'no'. Would you like personalized academic suggestions?"
        
        else:
            # Handle any other messages
            user_lower = user_msg.lower()
            if any(word in user_lower for word in ['hi', 'hello', 'hey']):
                return f"Hello again {conv['name']}! How can I help you?"
            elif any(word in user_lower for word in ['thanks', 'thank you']):
                return f"You're welcome {conv['name']}! Good luck with your studies! 🎓"
            elif any(word in user_lower for word in ['help', 'suggestion', 'advice']):
                return f"I can help with study techniques, time management, and academic planning. What specifically do you need, {conv['name']}?"
            else:
                return f"I'm here to help with academic suggestions, {conv['name']}. You can ask about study tips or specific improvements!"

# Initialize chat advisor
chat_advisor = ChatAdvisor()
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
# ==================== ENHANCED CHAT ADVISOR (FIXED FOR DATA CHECK) ====================
# ==================== ENHANCED CHAT ADVISOR (SIMPLIFIED & FIXED) ====================
class EnhancedChatAdvisor:
    def __init__(self):
        self.conversations = {}
    
    def generate_personalized_summary(self, student_data, name="Student"):
        """Generate a personalized summary with user's actual data"""
        analysis = advisor_model.analyze_student_profile(student_data)
        
        summary = f"📊 **Academic Summary for {name}**\n\n"
        summary += "🎯 **Your Performance Overview:**\n"
        summary += f"• **CGPA**: {student_data['total_cgpa']}/10\n"
        summary += f"• **Attendance**: {student_data['attendance']}%\n"
        summary += f"• **Study Hours**: {student_data['study_hours']} hrs/week\n"
        summary += f"• **Backlogs**: {student_data['backlogs']}\n"
        summary += f"• **Competitions**: {'Yes' if student_data['competitions'] else 'No'}\n"
        summary += f"• **Projects/Internships**: {'Yes' if student_data['projects_internships'] else 'No'}\n"
        summary += f"• **Confidence Level**: {student_data['confidence_level']}/10\n\n"
        
        if analysis['key_strengths']:
            summary += "✅ **Your Strengths:**\n"
            for strength in analysis['key_strengths'][:3]:
                summary += f"• {strength}\n"
            summary += "\n"
        
        if analysis['critical_areas']:
            summary += "🎯 **Focus Areas for Improvement:**\n"
            for area in analysis['critical_areas'][:3]:
                summary += f"• {area}\n"
            summary += "\n"
        
        summary += "💡 **Quick Action Plan:**\n"
        cgpa = student_data['total_cgpa']
        if cgpa < 8.0:
            summary += f"• Target CGPA: 8.0+ (Current: {cgpa}/10)\n"
        else:
            summary += f"• Maintain your excellent CGPA of {cgpa}/10\n"
        
        attendance = student_data['attendance']
        if attendance < 85:
            summary += f"• Improve attendance to 90%+ (Current: {attendance}%)\n"
        else:
            summary += f"• Great attendance at {attendance}%\n"
        
        study_hours = student_data['study_hours']
        if study_hours < 20:
            summary += f"• Increase study hours to 25+/week (Current: {study_hours} hrs)\n"
        else:
            summary += f"• Good study routine of {study_hours} hrs/week\n"
        
        backlogs = student_data['backlogs']
        if backlogs > 0:
            summary += f"• Clear {backlogs} backlog(s) this semester\n"
        else:
            summary += "• No backlogs - excellent!\n"
        
        if student_data['competitions'] == 0:
            summary += "• Participate in coding competitions\n"
        if student_data['projects_internships'] == 0:
            summary += "• Start building projects portfolio\n"
        
        summary += f"\n🎯 **Predicted Performance**: {student_data.get('predicted_class', 'Average')}\n"
        summary += "🚀 **Next Level**: " + advisor_model._get_target_performance(student_data.get('predicted_class', 'Average'))
        
        return summary
    
    def get_category_response(self, category):
        """Get detailed responses for each main category"""
        responses = {
            'summary': "summary",
            'academic performance analysis': """
📊 **Academic Performance Analysis & Improvement**

**Key Areas to Focus On:**
• **CGPA Improvement**: Target 8.0+ for better opportunities
• **Attendance Management**: Maintain 85%+ for better learning
• **Study Hours Optimization**: 25+ hours weekly with effective techniques
• **Backlog Clearance**: Strategic approach to clear pending subjects
• **Subject Balance**: Equal focus on theory and practical subjects

**Action Plan:**
1. Identify 2 weakest subjects for focused improvement
2. Create weekly study schedule with time slots
3. Use active recall and spaced repetition techniques
4. Regular self-assessment through mock tests
5. Seek faculty guidance for difficult topics
""",
            'study techniques time management': """
🎯 **Study Techniques & Time Management**

**Effective Study Methods:**
• **Pomodoro Technique**: 25min study + 5min break (4 cycles then long break)
• **Active Recall**: Test yourself instead of re-reading
• **Spaced Repetition**: Review at intervals (1d, 3d, 1w, 2w)
• **Feynman Technique**: Teach concepts in simple terms
• **Mind Mapping**: Visual organization of complex topics
""",
            'exam preparation strategies': """
📖 **Exam Preparation Strategies**

**3-Phase Preparation Plan:**
**Phase 1: Foundation (4-6 weeks before)**
• Complete syllabus reading
• Create chapter-wise notes
• Identify important topics

**Phase 2: Intensive Practice (2-3 weeks before)**
• Solve previous years' papers
• Chapter-wise mock tests
• Focus on weak areas

**Phase 3: Revision (Last week)**
• Quick revision of notes
• Formula/theorem practice
• Time management practice
""",
            'career guidance placements': """
💼 **Career Guidance & Placements**

**Placement Preparation Roadmap:**
• **Technical Skills**: DSA, OOPs, DBMS, OS
• **Practice**: LeetCode, HackerRank, CodeChef  
• **Projects**: 2-3 good projects with GitHub portfolio
• **Soft Skills**: Group Discussion, HR interview preparation
• **Resume**: Build with achievements and tailor for each company
""",
            'mental health motivation': """
😌 **Mental Health & Motivation**

**Stress Management:**
• Regular exercise (30min daily)
• 7-8 hours quality sleep  
• Healthy diet with proper hydration
• Mindfulness meditation (10min daily)
• Breaks and hobbies for relaxation
""",
            'campus life balance': """
🌿 **Campus Life & Balance**

**Extracurricular Activities:**
• Join clubs related to your interests
• Participate in college events
• Take leadership roles
• Build network with seniors and professors
• Maintain work-life balance
"""
        }
        return responses.get(category.lower(), "I can help you with that! Please ask more specifically.")
    
    def get_quick_actions(self):
        """Get quick action buttons for the bottom"""
        return [
            {'text': '📋 Get My Summary', 'query': 'summary'},
            {'text': '📊 Academic Analysis', 'query': 'academic performance analysis'},
            {'text': '🎯 Study Techniques', 'query': 'study techniques time management'},
            {'text': '📖 Exam Preparation', 'query': 'exam preparation strategies'},
            {'text': '💼 Career Guidance', 'query': 'career guidance placements'},
            {'text': '😌 Mental Health', 'query': 'mental health motivation'},
            {'text': '🌿 Campus Life', 'query': 'campus life balance'}
        ]

    def handle_message(self, session_id, user_message, student_data=None):
        # Initialize conversation if not exists
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'step': 'greeting',
                'name': None
            }

        conv = self.conversations[session_id]
        user_msg = user_message.strip()
        user_lower = user_msg.lower()
        
        # ========== FIRST CHANGE: NORMAL CONVERSATIONAL BOT ==========
        # Handle normal conversation - how to use, what, where, fields, doubts
        if any(word in user_lower for word in ['hi', 'hello', 'hey', 'greetings']):
            return "Hello! I'm your academic advisor. How can I help you today?"
        
        if any(word in user_lower for word in ['bye', 'goodbye', 'exit', 'quit', 'end chat']):
            name = conv.get('name', 'Student')
            return f"Goodbye {name}! Feel free to come back anytime for academic advice. Good luck with your studies! 🎓"
        
        if any(word in user_lower for word in ['thanks', 'thank you']):
            return "You're welcome! Happy to help! 😊"
        
        # How to use, what, where, fields, doubts
        if any(word in user_lower for word in ['how to use', 'how use', 'how this works']):
            return "I'm an academic advisor bot. You can ask me about:\n• Study techniques\n• Time management\n• Subject-specific strategies\n• Exam preparation\n• Career guidance\n• Mental health tips\n• Campus life balance\n\nJust ask your question!"
        
        if any(word in user_lower for word in ['what', 'what is']):
            if 'cgpa' in user_lower:
                return "📊 **CGPA** - Cumulative Grade Point Average (0-10 scale) shows your overall academic performance."
            elif 'attendance' in user_lower:
                return "📅 **Attendance** - Your class attendance percentage. 85%+ is recommended for better learning."
            elif 'study hours' in user_lower:
                return "⏰ **Study Hours** - Weekly study time outside class. 25+ hours is optimal for good performance."
            elif 'backlog' in user_lower:
                return "🔧 **Backlogs** - Subjects pending clearance. Zero backlogs is ideal for academic progress."
            elif 'competition' in user_lower:
                return "🏆 **Competitions** - Participation in coding/hackathon events shows practical skills."
            elif 'project' in user_lower or 'internship' in user_lower:
                return "💼 **Projects/Internships** - Hands-on experience valuable for placements and higher studies."
            elif 'confidence' in user_lower:
                return "💪 **Confidence Level** - Self-assessment of your academic confidence (1-10 scale)."
            else:
                return "I can explain academic terms like CGPA, attendance, study hours, backlogs, competitions, projects, and confidence levels. What would you like to know?"
        
        if any(word in user_lower for word in ['where', 'where to']):
            if 'study' in user_lower:
                return "📚 **Where to study**: Library, quiet classroom, study room, or any distraction-free environment that works for you!"
            elif 'help' in user_lower:
                return "🆘 **Where to get help**: Faculty during office hours, college counseling center, study groups, online forums, or academic support services."
            else:
                return "I can guide you on where to study, where to get academic help, where to find resources, etc. What specifically?"
        
        if any(word in user_lower for word in ['doubt', 'question', 'confused']):
            return "I can help clarify doubts about:\n• Study techniques\n• Time management\n• Exam preparation\n• Career choices\n• Subject-specific questions\n• Academic planning\n\nWhat's your specific doubt?"
        
        # ========== SECOND CHANGE: IF USER ASKS DIFFERENT QUESTIONS ==========
        # Check if the question matches our knowledge base keywords
        knowledge_keywords = [
            'study', 'technique', 'time', 'management', 'exam', 'preparation', 
            'career', 'placement', 'mental', 'health', 'motivation', 'campus',
            'cgpa', 'attendance', 'backlog', 'project', 'internship', 'confidence',
            'academic', 'performance', 'analysis', 'summary'
        ]
        
        # If user asks something completely different from our knowledge base
        if not any(keyword in user_lower for keyword in knowledge_keywords):
            return "Good answer! I'm here to help with academic advice whenever you need it. 😊"
        
        # ========== REST OF THE ORIGINAL CODE (UNCHANGED) ==========
        # Check if student_data exists and has valid prediction data
        has_student_data = False
        if student_data and isinstance(student_data, dict):
            # Check if we have predicted_class (which means form was submitted and predictions generated)
            has_prediction_result = 'predicted_class' in student_data and student_data['predicted_class'] is not None
            
            # Check if we have the essential academic data
            has_academic_data = ('total_cgpa' in student_data and 
                               student_data.get('total_cgpa') is not None and
                               student_data.get('total_cgpa') != '')
            
            has_student_data = has_prediction_result or has_academic_data

        print(f"DEBUG: has_student_data = {has_student_data}")
        if student_data:
            print(f"DEBUG: student_data keys = {student_data.keys()}")
            print(f"DEBUG: predicted_class = {student_data.get('predicted_class')}")
            print(f"DEBUG: total_cgpa = {student_data.get('total_cgpa')}")

        # ========== USER HAS DATA - SHOW PERSONALIZED SUGGESTIONS ==========
        if has_student_data:
            # Check for category-specific queries
            category_responses = {
                'academic performance analysis': self.get_category_response('academic performance analysis'),
                'study techniques time management': self.get_category_response('study techniques time management'),
                'exam preparation strategies': self.get_category_response('exam preparation strategies'),
                'career guidance placements': self.get_category_response('career guidance placements'),
                'mental health motivation': self.get_category_response('mental health motivation'),
                'campus life balance': self.get_category_response('campus life balance')
            }
            
            for category, response in category_responses.items():
                if any(word in user_lower for word in category.split()):
                    return response
            
            # Handle summary request
            if any(word in user_lower for word in ['summary', 'my details', 'my profile', 'table', 'overview']):
                name = conv.get('name', 'Student')
                return self.generate_personalized_summary(student_data, name)
            
            # NORMAL CONVERSATION FLOW (user has data)
            if conv['step'] == 'greeting':
                conv['step'] = 'get_name'
                return "🎉 Hello! I'm your academic advisor. I can see you have your academic performance results ready! What's your name?"
            
            elif conv['step'] == 'get_name':
                if len(user_msg) < 2:
                    return "Please enter a valid name:"
                conv['name'] = user_msg
                conv['step'] = 'show_options'
                return f"Nice to meet you, {conv['name']}! ✅ I have analyzed your academic data. Would you like me to provide personalized suggestions based on your '{student_data.get('predicted_class', 'Average')}' performance? (yes/no)"
            
            elif conv['step'] == 'show_options':
                if any(word in user_lower for word in ['yes', 'yeah', 'sure', 'ok', 'yep']):
                    # Generate personalized advice
                    advice = advisor_model.generate_advice(student_data, student_data.get('predicted_class', 'Average'))
                    conv['step'] = 'completed'
                    return f"Great! Here are my personalized suggestions for you, {conv['name']}:\n\n{advice}"
                elif any(word in user_lower for word in ['no', 'not', 'nope', 'later']):
                    conv['step'] = 'completed'
                    return f"No problem {conv['name']}! Feel free to ask me about study tips anytime, or use the quick action buttons below for specific advice."
                else:
                    return "Please answer with 'yes' or 'no'. Would you like personalized academic suggestions based on your data?"
            
            elif conv['step'] == 'completed':
                # Search knowledge base for academic queries
                results = help_system.search_knowledge(user_msg)
                if results:
                    response = f"Here's what I found about '{user_msg}':\n\n"
                    for i, result in enumerate(results[:2]):
                        response += f"**{result['topic']}** ({result['category']})\n"
                        response += f"{result['content']}\n\n"
                    return response
                else:
                    return f"I'm here to help with academic suggestions, {conv['name']}! You can ask about study tips or use the quick action buttons for specific advice!"
            
            else:
                return f"I'm here to help with academic suggestions, {conv['name']}! You can ask about study tips or use the quick action buttons for specific advice!"

        # ========== USER HAS NO DATA - ASK TO FILL FORM FIRST ==========
        else:
            # NO DATA - Show message to fill form first
            if conv['step'] == 'greeting':
                conv['step'] = 'get_name'
                return "👋 Hello! I'm your AI Academic Advisor. To get started, please fill out the form on the left side and click 'Analyze Performance' to get your predictions first. Then I can provide personalized suggestions! What's your name?"
            
            elif conv['step'] == 'get_name':
                if len(user_msg) < 2:
                    return "Please enter a valid name:"
                conv['name'] = user_msg
                conv['step'] = 'show_options'
                return f"Nice to meet you, {conv['name']}! 📊 **Important:** Please fill out the form on the left side first and click 'Analyze Performance' to get your predictions. Once you have your results, I can provide personalized academic suggestions! For now, I can help with general study advice. What would you like to know?"

            elif conv['step'] == 'show_options':
                # No data available - guide user to fill form
                if any(word in user_lower for word in ['form', 'fill', 'data', 'predict', 'analysis']):
                    return f"📝 **Form Instructions:**\n1. Fill all fields in the form on the left\n2. Click 'Analyze Performance' button\n3. Get your prediction results\n4. Then I can provide personalized suggestions!\n\nWhat specific field do you need help understanding, {conv['name']}?"
                elif any(word in user_lower for word in ['cgpa', 'gpa', 'grade']):
                    return "📊 **CGPA Explanation:** Your Cumulative Grade Point Average (0-10 scale) shows your overall academic performance. Higher CGPA (8.0+) indicates strong academic foundation."
                elif any(word in user_lower for word in ['attendance']):
                    return "📅 **Attendance:** Regular attendance (85%+) ensures you don't miss important concepts and maintains good faculty rapport."
                elif any(word in user_lower for word in ['study', 'hours']):
                    return "⏰ **Study Hours:** Weekly study time. 25+ hours is optimal for good academic performance."
                elif any(word in user_lower for word in ['backlog', 'arrear']):
                    return "🔧 **Backlogs:** Number of subjects pending clearance. Zero backlogs is ideal for academic progress."
                elif any(word in user_lower for word in ['competition', 'project', 'internship']):
                    return "🚀 **Competitions/Projects:** Participation shows practical skills and initiative. Valuable for placements and higher studies."
                elif any(word in user_lower for word in ['confidence']):
                    return "💪 **Confidence Level:** Self-assessment of your academic confidence (1-10). Higher confidence often correlates with better performance."
                else:
                    # If user asks something different, use the "Good answer" response
                    return "Good answer! I'm here to help with academic advice whenever you need it. 😊"
            
            else:
                # Handle general queries without data
                if any(word in user_lower for word in ['hi', 'hello', 'hey']):
                    return f"Hello again {conv['name']}! Remember to fill the form first to get personalized advice. How can I help?"
                elif any(word in user_lower for word in ['thanks', 'thank you']):
                    return f"You're welcome {conv['name']}! Good luck with your studies! 🎓"
                elif any(word in user_lower for word in ['help', 'suggestion', 'advice']):
                    return f"📊 **First Step Needed:** Please fill out the form on the left side and click 'Analyze Performance' to get your predictions. Then I can provide personalized academic suggestions! What specific help do you need with the form, {conv['name']}?"
                else:
                    # If user asks something different, use the "Good answer" response
                    return "Good answer! I'm here to help with academic advice whenever you need it. 😊"

# Replace the existing chat advisor with the fixed version
chat_advisor = EnhancedChatAdvisor()
# ==================== YOUR EXISTING FLASK APP ====================
app = Flask(__name__, template_folder='student_performance_dnn/templates')
app.secret_key = 'your_secret_key_here'

scaler = joblib.load("student_performance_dnn/production_model/scaler.pkl")
dnn_model = load_model("student_performance_dnn/production_model/student_performance_model.keras")
label_encoder = joblib.load("student_performance_dnn/production_model/label_encoder.pkl")

# --- CORRECT Feature order that matches your NEW SCALER (8 features) ---
SCALER_FEATURES = [
    'total_cgpa', 
    'attendance', 
    'study_hours', 
    'backlogs', 
    'competitions', 
    'projects_internships', 
    'prevsem_cgpa',
    'confidence_level'  # INCLUDED in new model
]

# Get class labels from your label encoder
CLASS_LABELS = list(label_encoder.classes_)

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

            # Store student data for chat
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

# ==================== NEW CHAT ROUTES ====================
@app.route('/start_chat', methods=['POST'])
def start_chat():
    try:
        session_id = request.json.get('session_id', 'default')
        student_data = session.get('student_data')
        
        print(f"CHAT DEBUG: Starting chat for session {session_id}")
        print(f"CHAT DEBUG: Student data available: {bool(student_data)}")
        
        # AUTO-RESET: If we have new student data, reset the conversation
        if student_data and session_id in chat_advisor.conversations:
            # Keep only the name if it exists, reset everything else
            old_name = chat_advisor.conversations[session_id].get('name')
            chat_advisor.conversations[session_id] = {
                'step': 'greeting',
                'name': old_name,
                'awaiting_topic': False
            }
        
        response = chat_advisor.handle_message(session_id, "", student_data)
        return jsonify({'response': response})
    except Exception as e:
        print(f"CHAT ERROR: {str(e)}")
        return jsonify({'response': f"I'm having trouble starting the chat. Please try again. Error: {str(e)}"})

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        session_id = request.json.get('session_id', 'default')
        user_message = request.json.get('message', '')
        student_data = session.get('student_data')
        
        print(f"CHAT DEBUG: Message from {session_id}: {user_message}")
        print(f"CHAT DEBUG: Student data: {student_data}")
        
        response = chat_advisor.handle_message(session_id, user_message, student_data)
        return jsonify({'response': response})
    except Exception as e:
        print(f"CHAT ERROR: {str(e)}")
        return jsonify({'response': "Sorry, I encountered an error. Please try again."})

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session_id = request.json.get('session_id', 'default')
    if session_id in chat_advisor.conversations:
        del chat_advisor.conversations[session_id]
    return jsonify({'status': 'success'})

# ==================== CRITICAL FIX: ADD CLEAR SESSION ROUTE ====================
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)