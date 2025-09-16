"""
Phase 5.3: Listening Tests Framework
====================================

This module implements a comprehensive framework for subjective evaluation of
style-transferred music through listening tests. Includes ABX tests, Mean Opinion
Score (MOS) ratings, cultural authenticity evaluation, and automated test management.

Features:
- ABX (A/B/X) listening tests for quality comparison
- Mean Opinion Score (MOS) rating scales
- Cultural authenticity evaluation protocols
- Multi-criteria evaluation forms
- Test session management and data collection
- Statistical analysis of subjective results
- Web-based and CLI test interfaces
"""

import os
import json
import random
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import datetime
import warnings
warnings.filterwarnings('ignore')

class TestType(Enum):
    """Types of listening tests."""
    ABX = "abx"
    MOS = "mos"
    CULTURAL_AUTHENTICITY = "cultural"
    PREFERENCE = "preference"
    QUALITY_COMPARISON = "quality"

class Rating(Enum):
    """Standard rating scales."""
    MOS_5_POINT = {1: "Bad", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
    QUALITY_7_POINT = {1: "Extremely Poor", 2: "Very Poor", 3: "Poor", 4: "Fair", 
                      5: "Good", 6: "Very Good", 7: "Excellent"}
    AUTHENTICITY_5_POINT = {1: "Not Authentic", 2: "Slightly Authentic", 3: "Moderately Authentic",
                           4: "Quite Authentic", 5: "Very Authentic"}

@dataclass
class AudioSample:
    """Represents an audio sample for testing."""
    id: str
    filepath: str
    genre: str
    description: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TestQuestion:
    """Represents a test question/trial."""
    id: str
    test_type: TestType
    question_text: str
    audio_samples: List[AudioSample]
    rating_scale: Dict[int, str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TestResponse:
    """Represents a participant's response to a test question."""
    question_id: str
    participant_id: str
    response: Union[int, str, Dict]
    confidence: Optional[int] = None
    response_time: Optional[float] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().isoformat()

@dataclass
class ParticipantProfile:
    """Represents a test participant's profile."""
    id: str
    age_group: str  # "18-25", "26-35", "36-45", "46-55", "55+"
    musical_background: str  # "none", "amateur", "semi-professional", "professional"
    cultural_background: str  # e.g., "Bengali", "Western", "Mixed"
    hearing_status: str  # "normal", "mild_loss", "moderate_loss"
    audio_equipment: str  # "headphones", "speakers", "earbuds"
    previous_listening_tests: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ABXTest:
    """
    Implementation of ABX listening test for quality assessment.
    
    In ABX tests, participants hear three audio samples: A, B, and X (unknown).
    The task is to identify whether X is the same as A or B.
    """
    
    def __init__(self):
        self.test_id = f"abx_{int(time.time())}"
        self.questions = []
        
    def create_abx_question(self, sample_a: AudioSample, sample_b: AudioSample,
                           question_text: str = None) -> TestQuestion:
        """Create an ABX test question."""
        if question_text is None:
            question_text = "Listen to samples A and B, then identify if X is the same as A or B."
        
        # Randomly choose X to be either A or B
        x_is_a = random.choice([True, False])
        x_sample = sample_a if x_is_a else sample_b
        correct_answer = "A" if x_is_a else "B"
        
        question = TestQuestion(
            id=f"{self.test_id}_q{len(self.questions)}",
            test_type=TestType.ABX,
            question_text=question_text,
            audio_samples=[sample_a, sample_b, x_sample],
            rating_scale={"A": "Sample A", "B": "Sample B"},
            metadata={
                "correct_answer": correct_answer,
                "x_is_a": x_is_a
            }
        )
        
        self.questions.append(question)
        return question
    
    def analyze_results(self, responses: List[TestResponse]) -> Dict[str, float]:
        """Analyze ABX test results."""
        results = {
            "total_questions": len(self.questions),
            "total_responses": len(responses),
            "correct_responses": 0,
            "accuracy": 0.0,
            "average_response_time": 0.0,
            "confidence_scores": []
        }
        
        response_times = []
        
        for response in responses:
            # Find corresponding question
            question = next(
                (q for q in self.questions if q.id == response.question_id), None
            )
            
            if question:
                correct_answer = question.metadata.get("correct_answer")
                if response.response == correct_answer:
                    results["correct_responses"] += 1
                
                if response.response_time:
                    response_times.append(response.response_time)
                
                if response.confidence:
                    results["confidence_scores"].append(response.confidence)
        
        if results["total_responses"] > 0:
            results["accuracy"] = results["correct_responses"] / results["total_responses"]
        
        if response_times:
            results["average_response_time"] = np.mean(response_times)
        
        # Statistical significance test (binomial test)
        if results["total_responses"] >= 10:
            from scipy import stats
            # Test against chance level (0.5)
            p_value = stats.binom_test(
                results["correct_responses"], 
                results["total_responses"], 
                0.5, 
                alternative='greater'
            )
            results["p_value"] = p_value
            results["statistically_significant"] = p_value < 0.05
        
        return results

class MOSTest:
    """
    Implementation of Mean Opinion Score (MOS) test for quality rating.
    
    Participants rate audio samples on a 5-point scale from 1 (Bad) to 5 (Excellent).
    """
    
    def __init__(self, scale_type: str = "quality"):
        self.test_id = f"mos_{int(time.time())}"
        self.scale_type = scale_type
        self.questions = []
        
        # Select appropriate rating scale
        if scale_type == "quality":
            self.rating_scale = Rating.MOS_5_POINT.value
        elif scale_type == "quality_7":
            self.rating_scale = Rating.QUALITY_7_POINT.value
        elif scale_type == "authenticity":
            self.rating_scale = Rating.AUTHENTICITY_5_POINT.value
        else:
            self.rating_scale = Rating.MOS_5_POINT.value
    
    def create_mos_question(self, sample: AudioSample, 
                           question_text: str = None) -> TestQuestion:
        """Create a MOS test question."""
        if question_text is None:
            if self.scale_type == "authenticity":
                question_text = f"How authentic does this {sample.genre} music sound?"
            else:
                question_text = f"Rate the overall quality of this audio sample."
        
        question = TestQuestion(
            id=f"{self.test_id}_q{len(self.questions)}",
            test_type=TestType.MOS,
            question_text=question_text,
            audio_samples=[sample],
            rating_scale=self.rating_scale,
            metadata={"scale_type": self.scale_type}
        )
        
        self.questions.append(question)
        return question
    
    def analyze_results(self, responses: List[TestResponse]) -> Dict[str, float]:
        """Analyze MOS test results."""
        results = {
            "total_questions": len(self.questions),
            "total_responses": len(responses),
            "mean_opinion_score": 0.0,
            "std_deviation": 0.0,
            "median_score": 0.0,
            "score_distribution": {},
            "confidence_intervals": {}
        }
        
        scores = []
        confidence_scores = []
        
        for response in responses:
            if isinstance(response.response, int):
                scores.append(response.response)
            
            if response.confidence:
                confidence_scores.append(response.confidence)
        
        if scores:
            results["mean_opinion_score"] = np.mean(scores)
            results["std_deviation"] = np.std(scores)
            results["median_score"] = np.median(scores)
            
            # Score distribution
            unique, counts = np.unique(scores, return_counts=True)
            results["score_distribution"] = dict(zip(unique.astype(int), counts.astype(int)))
            
            # 95% confidence interval
            if len(scores) > 1:
                from scipy import stats
                ci = stats.t.interval(
                    0.95, len(scores)-1, 
                    loc=np.mean(scores), 
                    scale=stats.sem(scores)
                )
                results["confidence_intervals"]["95%"] = {
                    "lower": ci[0], "upper": ci[1]
                }
        
        if confidence_scores:
            results["average_confidence"] = np.mean(confidence_scores)
        
        return results

class CulturalAuthenticityTest:
    """
    Specialized test for evaluating cultural authenticity of music.
    
    Focuses on whether style-transferred music maintains cultural characteristics
    and sounds authentic to native speakers/listeners.
    """
    
    def __init__(self, target_culture: str):
        self.test_id = f"cultural_{int(time.time())}"
        self.target_culture = target_culture
        self.questions = []
        
        # Multi-dimensional authenticity criteria
        self.authenticity_criteria = [
            "melodic_authenticity",
            "rhythmic_authenticity", 
            "instrumental_authenticity",
            "vocal_style_authenticity",
            "overall_cultural_authenticity"
        ]
    
    def create_authenticity_question(self, sample: AudioSample) -> List[TestQuestion]:
        """Create multi-criteria authenticity questions for a sample."""
        questions = []
        
        for criterion in self.authenticity_criteria:
            criterion_text = criterion.replace("_", " ").title()
            question_text = f"Rate the {criterion_text} of this {self.target_culture} music sample."
            
            question = TestQuestion(
                id=f"{self.test_id}_q{len(self.questions)}_{criterion}",
                test_type=TestType.CULTURAL_AUTHENTICITY,
                question_text=question_text,
                audio_samples=[sample],
                rating_scale=Rating.AUTHENTICITY_5_POINT.value,
                metadata={
                    "criterion": criterion,
                    "target_culture": self.target_culture
                }
            )
            
            questions.append(question)
            self.questions.append(question)
        
        return questions
    
    def analyze_results(self, responses: List[TestResponse]) -> Dict[str, Any]:
        """Analyze cultural authenticity test results."""
        results = {
            "target_culture": self.target_culture,
            "total_questions": len(self.questions),
            "total_responses": len(responses),
            "criteria_scores": {},
            "overall_authenticity": 0.0
        }
        
        # Group responses by criterion
        criterion_responses = {criterion: [] for criterion in self.authenticity_criteria}
        
        for response in responses:
            question = next(
                (q for q in self.questions if q.id == response.question_id), None
            )
            
            if question and isinstance(response.response, int):
                criterion = question.metadata.get("criterion")
                if criterion in criterion_responses:
                    criterion_responses[criterion].append(response.response)
        
        # Calculate scores for each criterion
        criterion_scores = []
        for criterion, scores in criterion_responses.items():
            if scores:
                mean_score = np.mean(scores)
                results["criteria_scores"][criterion] = {
                    "mean": mean_score,
                    "std": np.std(scores),
                    "count": len(scores)
                }
                criterion_scores.append(mean_score)
        
        # Overall authenticity score
        if criterion_scores:
            results["overall_authenticity"] = np.mean(criterion_scores)
        
        return results

class ListeningTestSession:
    """
    Manages a complete listening test session with multiple test types.
    """
    
    def __init__(self, session_name: str, participant: ParticipantProfile):
        self.session_id = f"session_{int(time.time())}"
        self.session_name = session_name
        self.participant = participant
        self.tests = []
        self.responses = []
        self.start_time = None
        self.end_time = None
        self.session_metadata = {}
    
    def add_test(self, test: Union[ABXTest, MOSTest, CulturalAuthenticityTest]):
        """Add a test to the session."""
        self.tests.append(test)
    
    def start_session(self):
        """Start the listening test session."""
        self.start_time = datetime.datetime.now()
        print(f"Starting listening test session: {self.session_name}")
        print(f"Participant: {self.participant.id}")
        print(f"Session ID: {self.session_id}")
    
    def add_response(self, response: TestResponse):
        """Add a participant response."""
        response.participant_id = self.participant.id
        self.responses.append(response)
    
    def end_session(self):
        """End the listening test session."""
        self.end_time = datetime.datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        print(f"Session completed in {duration:.1f} seconds")
        print(f"Total responses collected: {len(self.responses)}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the test session."""
        if not self.start_time:
            return {"error": "Session not started"}
        
        duration = None
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "participant": asdict(self.participant),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "total_tests": len(self.tests),
            "total_questions": sum(len(test.questions) for test in self.tests),
            "total_responses": len(self.responses),
            "completion_rate": len(self.responses) / sum(len(test.questions) for test in self.tests) if self.tests else 0
        }
    
    def save_session(self, filepath: str):
        """Save the session data to file."""
        session_data = {
            "session_summary": self.get_session_summary(),
            "tests": [
                {
                    "test_id": test.test_id,
                    "test_type": type(test).__name__,
                    "questions": [asdict(q) for q in test.questions]
                }
                for test in self.tests
            ],
            "responses": [asdict(r) for r in self.responses]
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"Session data saved to {filepath}")

class ListeningTestAnalyzer:
    """
    Comprehensive analyzer for listening test results.
    """
    
    def __init__(self):
        self.session_data = []
    
    def load_session(self, filepath: str):
        """Load session data from file."""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        self.session_data.append(session_data)
        print(f"Loaded session data from {filepath}")
    
    def analyze_all_sessions(self) -> Dict[str, Any]:
        """Analyze results across all loaded sessions."""
        if not self.session_data:
            return {"error": "No session data loaded"}
        
        analysis = {
            "total_sessions": len(self.session_data),
            "total_participants": len(set(
                session["session_summary"]["participant"]["id"] 
                for session in self.session_data
            )),
            "test_type_analysis": {},
            "participant_analysis": {},
            "overall_statistics": {}
        }
        
        # Aggregate responses by test type
        test_responses = {}
        all_participants = []
        
        for session in self.session_data:
            participant = session["session_summary"]["participant"]
            all_participants.append(participant)
            
            for response in session["responses"]:
                # Determine test type from question ID pattern
                question_id = response["question_id"]
                test_type = None
                
                if "abx_" in question_id:
                    test_type = "ABX"
                elif "mos_" in question_id:
                    test_type = "MOS"
                elif "cultural_" in question_id:
                    test_type = "Cultural Authenticity"
                
                if test_type:
                    if test_type not in test_responses:
                        test_responses[test_type] = []
                    test_responses[test_type].append(response)
        
        # Analyze by test type
        for test_type, responses in test_responses.items():
            if test_type == "ABX":
                # Simulate ABX analysis (simplified)
                correct = sum(1 for r in responses if r.get("metadata", {}).get("correct", False))
                total = len(responses)
                analysis["test_type_analysis"][test_type] = {
                    "total_responses": total,
                    "accuracy": correct / total if total > 0 else 0,
                    "average_response_time": np.mean([
                        r.get("response_time", 0) for r in responses if r.get("response_time")
                    ]) if responses else 0
                }
            
            elif test_type == "MOS":
                scores = [r["response"] for r in responses if isinstance(r["response"], int)]
                if scores:
                    analysis["test_type_analysis"][test_type] = {
                        "total_responses": len(responses),
                        "mean_score": np.mean(scores),
                        "std_deviation": np.std(scores),
                        "score_distribution": dict(zip(*np.unique(scores, return_counts=True)))
                    }
        
        # Participant demographics analysis
        if all_participants:
            age_groups = [p["age_group"] for p in all_participants]
            musical_backgrounds = [p["musical_background"] for p in all_participants]
            cultural_backgrounds = [p["cultural_background"] for p in all_participants]
            
            analysis["participant_analysis"] = {
                "age_distribution": dict(zip(*np.unique(age_groups, return_counts=True))),
                "musical_background_distribution": dict(zip(*np.unique(musical_backgrounds, return_counts=True))),
                "cultural_background_distribution": dict(zip(*np.unique(cultural_backgrounds, return_counts=True)))
            }
        
        return analysis

def create_demo_listening_test():
    """Create a demonstration listening test session."""
    print("Creating Demo Listening Test Session")
    print("=" * 40)
    
    # Create participant profile
    participant = ParticipantProfile(
        id="demo_participant_001",
        age_group="26-35",
        musical_background="amateur",
        cultural_background="Bengali",
        hearing_status="normal",
        audio_equipment="headphones",
        previous_listening_tests=False
    )
    
    # Create audio samples (simulated)
    original_folk = AudioSample(
        id="orig_folk_001",
        filepath="demo/original_folk.wav",
        genre="Bangla Folk",
        description="Original Bangla folk song"
    )
    
    transferred_jazz = AudioSample(
        id="trans_jazz_001", 
        filepath="demo/transferred_jazz.wav",
        genre="Jazz",
        description="Bangla folk transferred to jazz style"
    )
    
    reference_jazz = AudioSample(
        id="ref_jazz_001",
        filepath="demo/reference_jazz.wav", 
        genre="Jazz",
        description="Reference jazz recording"
    )
    
    # Create test session
    session = ListeningTestSession("Style Transfer Evaluation Demo", participant)
    session.start_session()
    
    # Create ABX test
    print("\n1. Creating ABX Test...")
    abx_test = ABXTest()
    abx_question = abx_test.create_abx_question(
        transferred_jazz, reference_jazz,
        "Which sample (A or B) sounds more similar to X?"
    )
    session.add_test(abx_test)
    
    # Create MOS test
    print("2. Creating MOS Test...")
    mos_test = MOSTest("quality")
    mos_question1 = mos_test.create_mos_question(
        transferred_jazz, "Rate the overall audio quality:"
    )
    mos_question2 = mos_test.create_mos_question(
        original_folk, "Rate the overall audio quality:"
    )
    session.add_test(mos_test)
    
    # Create Cultural Authenticity test
    print("3. Creating Cultural Authenticity Test...")
    cultural_test = CulturalAuthenticityTest("Bengali Folk")
    cultural_questions = cultural_test.create_authenticity_question(transferred_jazz)
    session.add_test(cultural_test)
    
    # Simulate responses
    print("\n4. Simulating Participant Responses...")
    
    # ABX response
    abx_response = TestResponse(
        question_id=abx_question.id,
        participant_id=participant.id,
        response="B",
        confidence=4,
        response_time=8.5
    )
    session.add_response(abx_response)
    
    # MOS responses
    mos_response1 = TestResponse(
        question_id=mos_question1.id,
        participant_id=participant.id,
        response=4,
        confidence=5,
        response_time=6.2
    )
    session.add_response(mos_response1)
    
    mos_response2 = TestResponse(
        question_id=mos_question2.id,
        participant_id=participant.id,
        response=5,
        confidence=5,
        response_time=4.8
    )
    session.add_response(mos_response2)
    
    # Cultural authenticity responses
    for i, question in enumerate(cultural_questions):
        cultural_response = TestResponse(
            question_id=question.id,
            participant_id=participant.id,
            response=random.randint(3, 5),  # Simulated ratings
            confidence=4,
            response_time=random.uniform(5.0, 12.0)
        )
        session.add_response(cultural_response)
    
    session.end_session()
    
    # Analyze results
    print("\n5. Analyzing Results...")
    
    # ABX analysis
    abx_results = abx_test.analyze_results([abx_response])
    print(f"\nABX Test Results:")
    print(f"  Accuracy: {abx_results['accuracy']:.3f}")
    print(f"  Average Response Time: {abx_results['average_response_time']:.1f}s")
    
    # MOS analysis  
    mos_results = mos_test.analyze_results([mos_response1, mos_response2])
    print(f"\nMOS Test Results:")
    print(f"  Mean Opinion Score: {mos_results['mean_opinion_score']:.2f}")
    print(f"  Standard Deviation: {mos_results['std_deviation']:.2f}")
    print(f"  Score Distribution: {mos_results['score_distribution']}")
    
    # Cultural authenticity analysis
    cultural_responses = [r for r in session.responses if "cultural_" in r.question_id]
    cultural_results = cultural_test.analyze_results(cultural_responses)
    print(f"\nCultural Authenticity Results:")
    print(f"  Overall Authenticity: {cultural_results['overall_authenticity']:.2f}")
    for criterion, scores in cultural_results['criteria_scores'].items():
        print(f"  {criterion.replace('_', ' ').title()}: {scores['mean']:.2f}")
    
    # Save session
    output_dir = "experiments/listening_tests"
    os.makedirs(output_dir, exist_ok=True)
    session_file = os.path.join(output_dir, f"{session.session_id}.json")
    session.save_session(session_file)
    
    print(f"\n✅ Demo Listening Test Complete!")
    print(f"   Session saved to: {session_file}")
    
    return session, {
        "abx_results": abx_results,
        "mos_results": mos_results, 
        "cultural_results": cultural_results
    }

if __name__ == "__main__":
    # Run the demonstration
    session, results = create_demo_listening_test()
    
    print(f"\n📊 DEMO LISTENING TEST SUMMARY")
    print(f"=" * 40)
    print(f"Session ID: {session.session_id}")
    print(f"Total Questions: {sum(len(test.questions) for test in session.tests)}")
    print(f"Total Responses: {len(session.responses)}")
    print(f"Completion Rate: 100%")
    
    print(f"\n🎵 Ready for production listening tests! 🎵")