from faker import Faker
import random, json, uuid, os

fake = Faker()

SKILLS = [
    "Python","Java","AWS","Docker","React","React Native","TensorFlow","PyTorch",
    "Kubernetes","GCP","Azure","SQL","Node","Go","Scala","Spark","Pandas",
    "scikit-learn","FastAPI","Flask","NLP","Computer Vision","MongoDB","PostgreSQL"
]
PROJECT_TEMPLATES = [
    ("Healthcare Dashboard","healthcare"),
    ("Medical Diagnosis Platform","healthcare"),
    ("E-commerce Platform","e-commerce"),
    ("Fraud Detection Service","fintech"),
    ("DevOps Automation","devops"),
    ("Education Analytics","education"),
    ("Gaming Leaderboard","gaming"),
    ("Risk Prediction System","fintech"),
    ("Claims Processing API","healthcare")
]

def rand_skills():
    n = random.randint(3,6)
    return random.sample(SKILLS, n)

def rand_projects():
    n = random.randint(1,3)
    choices = random.sample(PROJECT_TEMPLATES, n)
    return [c[0] for c in choices]

def generate(n=20):
    employees = []
    for _ in range(n):
        emp = {
            "id": str(uuid.uuid4()),
            "name": fake.name(),
            "skills": rand_skills(),
            "experience_years": random.randint(1,10),
            "projects": rand_projects(),
            "availability": random.choice(["available","not available"])
        }
        employees.append(emp)
    return {"employees": employees}

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    data = generate(20)
    with open("./data/employee_data.json","w",encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Generated ./data/employee_data.json")
