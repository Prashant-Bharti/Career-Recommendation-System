import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn.functional as F
import streamlit as st

# -----------------------
# Step 1: Large Diverse Dataset
# -----------------------
data = {
    'User_Input': [
        # Software Engineer
        "I love coding and solving complex software problems",
        "Programming and app development interest me",
        "I enjoy building software systems and debugging code",
        "Solving technical challenges through code excites me",
        "Creating backend services and scalable systems is thrilling",
        "Software architecture and clean code matter to me",

        # Psychologist
        "I like listening to people and helping with mental issues",
        "Mental health advocacy and therapy attract me",
        "Understanding emotions and counseling others is fulfilling",
        "Helping people with anxiety and depression interests me",
        "I enjoy psychology and understanding human behavior",

        # Politician
        "Public speaking and leadership come naturally to me",
        "I'm passionate about governance and making laws",
        "I want to impact society through political engagement",
        "Debating policies and representing communities excites me",
        "Social reforms and civic engagement attract me",

        # Graphic Designer
        "I enjoy creating digital artwork and visual content",
        "Designing logos, posters, and art is my passion",
        "I have a flair for aesthetics and digital creativity",
        "Illustrating ideas visually and working on branding interests me",
        "Making visually appealing designs excites me",

        # Data Analyst
        "I like interpreting data to find patterns",
        "Data-driven decision making is exciting to me",
        "I enjoy working with numbers, stats, and trends",
        "Analyzing data and building dashboards is fun",
        "Making sense of data through graphs and charts excites me",

        # Teacher
        "I love teaching and simplifying tough topics",
        "Helping others learn gives me satisfaction",
        "Explaining ideas to students makes me happy",
        "Developing lesson plans and interacting with students excites me",
        "Sharing knowledge and guiding students is my passion",

        # Scientist
        "I’m curious about experiments and discovery",
        "Working in a lab and testing theories fascinates me",
        "Research and scientific exploration interest me",
        "I want to solve mysteries of nature through science",
        "Innovating through experiments and publishing papers excites me",

        # Mechanical Engineer
        "Machines and how they work fascinate me",
        "Fixing mechanical systems and building tools is fun",
        "I enjoy working with gears, engines, and tools",
        "Creating and improving mechanical parts excites me",
        "I like designing mechanical solutions to real problems",

        # Film Director
        "I'm passionate about filmmaking and storytelling",
        "I want to direct movies and bring stories to life",
        "Creating films and leading a production team excites me",
        "Screenwriting and working with actors interests me",
        "Telling visual stories and managing shoots is my dream",

        # Lawyer
        "I love debates, justice, and solving legal issues",
        "Legal reasoning and structured arguments interest me",
        "I enjoy reading about laws and helping in legal matters",
        "Fighting for justice in court and giving legal advice excites me",
        "Law interpretation and litigation intrigue me"
    ],
    'Career': [
        'Software Engineer']*6 + ['Psychologist']*5 + ['Politician']*5 + ['Graphic Designer']*5 + 
        ['Data Analyst']*5 + ['Teacher']*5 + ['Scientist']*5 + ['Mechanical Engineer']*5 + 
        ['Film Director']*5 + ['Lawyer']*5
}

# -----------------------
# Step 2: Prepare Data
# -----------------------
df = pd.DataFrame(data)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Career'])

# -----------------------
# Step 3: Dataset Class
# -----------------------
class CareerDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }

# -----------------------
# Step 4: BERT-based Model
# -----------------------
class CareerModel(nn.Module):
    def __init__(self, num_classes):
        super(CareerModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x), pooled_output

# -----------------------
# Step 5: Model Setup (Instantiate model directly instead of loading)
# -----------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CareerModel(num_classes=len(df['Career'].unique()))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# -----------------------
# Step 6: Semantic Recommendation Function
# -----------------------
def recommend_semantic(text):
    input_enc = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64).to(device)
    with torch.no_grad():
        _, query_emb = model(input_enc['input_ids'], input_enc['attention_mask'])

    text_embeddings = []
    for t in df['User_Input']:
        enc = tokenizer(t, return_tensors='pt', truncation=True, padding='max_length', max_length=64).to(device)
        with torch.no_grad():
            _, emb = model(enc['input_ids'], enc['attention_mask'])
        text_embeddings.append(emb.cpu().numpy())

    text_embeddings = np.vstack(text_embeddings)
    query_emb = query_emb.cpu().numpy()
    similarities = F.cosine_similarity(torch.tensor(text_embeddings), torch.tensor(query_emb), dim=1)
    best_match = torch.argmax(similarities).item()
    return df['Career'][best_match]

# -----------------------
# Step 7: Streamlit Interface
# -----------------------
st.title("AI-Powered Career Recommendation System")
st.write("Enter your interests or a description of what you enjoy doing:")

user_input = st.text_area("Your Description", height=100)
if st.button("Suggest Career"):
    if user_input.strip() == "":
        st.warning("Please enter a description of your interests.")
    else:
        try:
            suggestion = recommend_semantic(user_input)
            st.success(f"Suggested Career: {suggestion}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
