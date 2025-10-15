import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class StudentPerformanceTester:
    def __init__(self):
        # Paths to your trained model and artifacts
        self.model_path = r"C:\Users\Sheema\OneDrive\Desktop\mini_project\student_performance_dnn\production_model\student_performance_model.keras"
        self.scaler_path = r"C:\Users\Sheema\OneDrive\Desktop\mini_project\student_performance_dnn\production_model\scaler.pkl"
        self.label_encoder_path = r"C:\Users\Sheema\OneDrive\Desktop\mini_project\student_performance_dnn\production_model\label_encoder.pkl"
        
        # Feature columns (must match training)
        self.feature_columns = [
            'total_cgpa', 'attendance', 'study_hours', 'backlogs', 
            'competitions', 'projects_internships', 'prevsem_cgpa', 'confidence_level'
        ]
        
        # Load model and artifacts
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_names = []
        
        self.load_model_and_artifacts()
    
    def load_model_and_artifacts(self):
        """Load the trained model and preprocessing artifacts"""
        try:
            print("🔧 Loading model and artifacts...")
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder = joblib.load(self.label_encoder_path)
            self.class_names = list(self.label_encoder.classes_)
            
            print("✅ Model loaded successfully!")
            print(f"🎯 Classes: {self.class_names}")
            print(f"📊 Model input shape: {self.model.input_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single(self, features_dict):
        """Predict performance for a single student"""
        try:
            # Convert dictionary to array in correct order
            features_array = np.array([[
                features_dict['total_cgpa'],
                features_dict['attendance'], 
                features_dict['study_hours'],
                features_dict['backlogs'],
                features_dict['competitions'],
                features_dict['projects_internships'],
                features_dict['prevsem_cgpa'],
                features_dict['confidence_level']
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            # Get all probabilities
            probabilities = {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(prediction[0])
            }
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None, None
    
    def test_predefined_cases(self):
        """Test with predefined test cases"""
        print("\n" + "="*70)
        print("🧪 TESTING WITH PREDEFINED CASES")
        print("="*70)
        
        test_cases = [
            {
                'name': 'Excellent Student',
                'features': {
                    'total_cgpa': 9.2, 'attendance': 95, 'study_hours': 6,
                    'backlogs': 0, 'competitions': 3, 'projects_internships': 2,
                    'prevsem_cgpa': 9.0, 'confidence_level': 9
                },
                'expected': 'Excellent'
            },
            {
                'name': 'Good Student', 
                'features': {
                    'total_cgpa': 7.8, 'attendance': 85, 'study_hours': 4,
                    'backlogs': 1, 'competitions': 2, 'projects_internships': 1,
                    'prevsem_cgpa': 7.5, 'confidence_level': 7
                },
                'expected': 'Good'
            },
            {
                'name': 'Average Student',
                'features': {
                    'total_cgpa': 6.5, 'attendance': 75, 'study_hours': 3,
                    'backlogs': 2, 'competitions': 1, 'projects_internships': 0,
                    'prevsem_cgpa': 6.2, 'confidence_level': 5
                },
                'expected': 'Average'
            },
            {
                'name': 'Below Average Student',
                'features': {
                    'total_cgpa': 5.0, 'attendance': 60, 'study_hours': 1,
                    'backlogs': 6, 'competitions': 0, 'projects_internships': 0,
                    'prevsem_cgpa': 4.8, 'confidence_level': 3
                },
                'expected': 'Below Average'
            },
            {
                'name': 'Borderline Excellent',
                'features': {
                    'total_cgpa': 8.8, 'attendance': 92, 'study_hours': 5,
                    'backlogs': 0, 'competitions': 2, 'projects_internships': 2,
                    'prevsem_cgpa': 8.5, 'confidence_level': 8
                },
                'expected': 'Excellent'
            },
            {
                'name': 'Borderline Good/Average',
                'features': {
                    'total_cgpa': 7.0, 'attendance': 78, 'study_hours': 3,
                    'backlogs': 1, 'competitions': 1, 'projects_internships': 0,
                    'prevsem_cgpa': 6.8, 'confidence_level': 6
                },
                'expected': 'Good'
            }
        ]
        
        results = []
        
        for case in test_cases:
            print(f"\n📝 {case['name']}:")
            print(f"   Input: CGPA={case['features']['total_cgpa']}, "
                  f"Attendance={case['features']['attendance']}%, "
                  f"Study={case['features']['study_hours']}hrs, "
                  f"Backlogs={case['features']['backlogs']}")
            
            predicted, confidence, probabilities = self.predict_single(case['features'])
            
            if predicted:
                status = "✅" if predicted == case['expected'] else "❌"
                print(f"   Expected: {case['expected']}")
                print(f"   Predicted: {predicted} {status}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Probabilities: {probabilities}")
                
                results.append({
                    'case': case['name'],
                    'expected': case['expected'],
                    'predicted': predicted,
                    'confidence': confidence,
                    'correct': predicted == case['expected']
                })
            else:
                print(f"   ❌ Prediction failed")
                results.append({
                    'case': case['name'],
                    'expected': case['expected'],
                    'predicted': 'FAILED',
                    'confidence': 0.0,
                    'correct': False
                })
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in results if r['correct'])
        total_cases = len(results)
        accuracy = correct_predictions / total_cases
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   ✅ Correct: {correct_predictions}/{total_cases}")
        print(f"   🎯 Accuracy: {accuracy:.1%}")
        
        return results
    
    def interactive_testing(self):
        """Interactive testing mode"""
        print("\n" + "="*70)
        print("🎮 INTERACTIVE TESTING MODE")
        print("="*70)
        
        while True:
            try:
                print("\nEnter student details (or 'quit' to exit):")
                
                # Get input from user
                features = {}
                for feature in self.feature_columns:
                    while True:
                        try:
                            value = input(f"   {feature.replace('_', ' ').title()}: ")
                            if value.lower() == 'quit':
                                return
                            features[feature] = float(value)
                            break
                        except ValueError:
                            print("     Please enter a valid number")
                
                # Make prediction
                print("\n🤔 Making prediction...")
                predicted, confidence, probabilities = self.predict_single(features)
                
                if predicted:
                    print(f"\n🎯 PREDICTION RESULT:")
                    print(f"   Performance: {predicted}")
                    print(f"   Confidence: {confidence:.3f}")
                    
                    # Show probability distribution
                    print(f"\n📊 Probability Distribution:")
                    for class_name, prob in probabilities.items():
                        print(f"   {class_name}: {prob:.3f}")
                    
                    # Interpretation
                    if confidence > 0.9:
                        print("   💪 High confidence prediction")
                    elif confidence > 0.7:
                        print("   ⚠️  Medium confidence prediction")
                    else:
                        print("   🔄 Low confidence prediction")
                
                else:
                    print("❌ Prediction failed!")
                
                print("\n" + "-"*50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def batch_test_from_dataset(self, dataset_path):
        """Test the model on a dataset file"""
        print(f"\n" + "="*70)
        print(f"📊 BATCH TESTING FROM DATASET")
        print("="*70)
        
        try:
            # Load dataset
            df = pd.read_excel(dataset_path)
            print(f"✅ Loaded dataset: {len(df)} samples")
            
            # Check if required columns exist
            required_cols = self.feature_columns + ['performance_category']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return
            
            # Test first 10 samples
            test_samples = df.head(10)
            results = []
            
            print(f"\nTesting first 10 samples:")
            print("-" * 60)
            
            for idx, row in test_samples.iterrows():
                features = {col: row[col] for col in self.feature_columns}
                actual = row['performance_category']
                
                predicted, confidence, _ = self.predict_single(features)
                
                status = "✅" if predicted == actual else "❌"
                print(f"Sample {idx+1}: Actual={actual}, Predicted={predicted} {status} (Confidence: {confidence:.3f})")
                
                results.append({
                    'sample': idx+1,
                    'actual': actual,
                    'predicted': predicted,
                    'confidence': confidence,
                    'correct': predicted == actual
                })
            
            # Calculate accuracy
            correct = sum(1 for r in results if r['correct'])
            accuracy = correct / len(results)
            
            print(f"\n📈 BATCH TEST RESULTS:")
            print(f"   ✅ Correct: {correct}/{len(results)}")
            print(f"   🎯 Accuracy: {accuracy:.1%}")
            
        except Exception as e:
            print(f"❌ Error in batch testing: {e}")

def main():
    print("🤖 STUDENT PERFORMANCE MODEL TESTER")
    print("=" * 70)
    
    # Initialize tester
    tester = StudentPerformanceTester()
    
    while True:
        print("\nSelect testing mode:")
        print("1. Predefined test cases")
        print("2. Interactive testing") 
        print("3. Batch test from dataset")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            tester.test_predefined_cases()
        elif choice == '2':
            tester.interactive_testing()
        elif choice == '3':
            dataset_path = r"C:\Users\Sheema\OneDrive\Desktop\mini_project\student_performance_dnn\data\dataset_balanced.xlsx"
            tester.batch_test_from_dataset(dataset_path)
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()