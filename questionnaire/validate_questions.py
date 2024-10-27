import pickle

class MentalHealthScorer:
    def __init__(self):
        # Scoring weights for each option position (0-3 index in options array)
        self.option_weights = {
            # For positively framed questions (higher is better)
            'positive': [10, 7, 4, 1],
            # For negatively framed questions (higher is worse)
            'negative': [1, 4, 7, 10]
        }
        
        # Define scoring rules for each category
        self.scoring_rules = {
            'Anxiety': {
                'question_types': ['positive', 'negative', 'positive', 'negative', 'positive'],
                'weights': [1, 1, 1, 1, 1]
            },
            'Depression': {
                'question_types': ['positive', 'positive', 'positive', 'positive', 'negative'],
                'weights': [1, 1, 1, 1, 1]
            },
            'Career Confusion': {
                'question_types': ['positive', 'positive', 'negative', 'positive', 'negative'],
                'weights': [1, 1, 1, 1, 1]
            },
            'Positive Outlook': {
                'question_types': ['positive', 'positive', 'positive', 'positive', 'positive'],
                'weights': [1, 1, 1, 1, 1]
            },
            'Health Anxiety': {
                'question_types': ['negative', 'negative', 'negative', 'positive', 'negative'],
                'weights': [1, 1, 1, 1, 1]
            },
            'Insomnia': {
                'question_types': ['positive', 'negative', 'negative', 'positive', 'negative'],
                'weights': [1, 1, 1, 1, 1]
            },
            'Stress': {
                'question_types': ['negative', 'positive', 'negative', 'positive', 'negative'],
                'weights': [1, 1, 1, 1, 1]
            },
            'Eating Disorder': {
                'question_types': ['positive', 'negative', 'negative', 'positive', 'negative'],
                'weights': [1, 1, 1, 1, 1]
            }
        }
        
        # Categories where higher scores indicate more severity
        self.negative_categories = {
            'Anxiety', 'Depression', 'Career Confusion', 'Health Anxiety',
            'Insomnia', 'Stress', 'Eating Disorder'
        }

    def calculate_score(self, responses, category):
        """
        Calculate the mental health score for a given category based on responses.
        
        Args:
            responses (list): List of integers (0-3) corresponding to selected options
            category (str): Category name to calculate score for
            
        Returns:
            int: Score between 1-10, or None if invalid category
        """
        if category not in self.scoring_rules:
            return None
            
        rule = self.scoring_rules[category]
        if len(responses) != len(rule['question_types']):
            raise ValueError(f"Expected {len(rule['question_types'])} responses for {category}")
            
        total_score = 0
        total_weight = 0
        
        # Calculate weighted score
        for response, q_type, weight in zip(
            responses,
            rule['question_types'],
            rule['weights']
        ):
            if not (0 <= response <= 3):
                raise ValueError("Responses must be between 0 and 3")
                
            score = self.option_weights[q_type][response]
            total_score += score * weight
            total_weight += weight
            
        # Normalize score to 1-10 scale
        normalized_score = (total_score / total_weight)
        
        # Invert score for positive categories
        if category not in self.negative_categories:
            normalized_score = 11 - normalized_score
            
        return round(normalized_score)

    def get_intensity_level(self, score):
        """
        Get a descriptive intensity level based on the numerical score.
        
        Args:
            score (int): Score between 1-10
            
        Returns:
            str: Descriptive intensity level
        """
        if score <= 3:
            return "Low"
        elif score <= 6:
            return "Moderate"
        elif score <= 8:
            return "High"
        else:
            return "Severe"

    def validate_responses(self, category, responses):
        """
        Validate that responses are correct for the given category.
        
        Args:
            category (str): Category name
            responses (list): List of response indices
            
        Returns:
            bool: True if valid, False otherwise
        """
        if category not in self.scoring_rules:
            return False
            
        rule = self.scoring_rules[category]
        if len(responses) != len(rule['question_types']):
            return False
            
        return all(0 <= r <= 3 for r in responses)


def main():
    scorer = MentalHealthScorer()

    with open('responses.pkl', 'rb') as file:
        data = pickle.load(file)
    
    for disorder, responses in data.items():
        print("disorder and response: ",disorder, responses)
        if scorer.validate_responses(disorder, responses):
            score = scorer.calculate_score(responses, disorder)
            intensity = scorer.get_intensity_level(score)
            print(f"{disorder} Score: {score}/10 - {intensity}\n")
        else:
            print(f"Invalid responses for {disorder}\n\n")    
    try:
        # score = scorer.calculate_score(responses, 'Anxiety')
        # intensity = scorer.get_intensity_level(score)
        # print(f"Anxiety Score: {score}/10")
        # print(f"Intensity Level: {intensity}")
        
        categories = ['Anxiety', 'Depression', 'Stress', 'Positive Outlook', 'Eating Disorder', 'Insomnia', 'Health Anxiety', 'Career Confusion', 'Mixed Reactions']
        for category in categories:
            if scorer.validate_responses(category, responses):
                score = scorer.calculate_score(responses, category)
                print(f"{category} Score: {score}/10 - {scorer.get_intensity_level(score)}")
                
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()