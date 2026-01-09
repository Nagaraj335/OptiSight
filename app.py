# Eye Disease Detection GUI Application
# Supports: AMD, Diabetic Retinopathy, Cataract, Glaucoma Detection

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
import subprocess

# Ensure models are downloaded (for Streamlit Cloud)
def ensure_models():
    """Download models from Git LFS if not present"""
    os.makedirs('outputs/models', exist_ok=True)
    
    model_files = [
        'outputs/models/best_model_resnet50.pth',
        'outputs/models/glaucoma_model.pth'
    ]
    
    missing = [f for f in model_files if not os.path.exists(f) or os.path.getsize(f) < 1000]
    
    if missing:
        try:
            subprocess.run(['git', 'lfs', 'pull'], check=False, capture_output=True)
        except:
            pass

ensure_models()

# Page configuration
st.set_page_config(
    page_title="Eye Disease Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .abnormal {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Model Configuration
AMD_CLASSES = ['AMD', 'Cataract', 'Diabetic Retinopathy', 'Normal']
GLAUCOMA_CLASSES = ['Normal', 'Glaucoma']
IMG_SIZE = 224

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class AMDClassifier(nn.Module):
    """AMD Classification model (matches training structure)"""
    def __init__(self, num_classes=4):
        super(AMDClassifier, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class GlaucomaClassifier(nn.Module):
    """Glaucoma Classification model (matches training structure)"""
    def __init__(self, num_classes=2):
        super(GlaucomaClassifier, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_amd_model():
    """Load the trained AMD detection model"""
    try:
        model = AMDClassifier(num_classes=4)
        
        checkpoint = torch.load('outputs/models/best_model_resnet50.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, checkpoint.get('val_acc', None)
    except Exception as e:
        st.error(f"Error loading AMD model: {e}")
        return None, None

@st.cache_resource
def load_glaucoma_model():
    """Load the trained Glaucoma detection model"""
    # Try multiple possible locations
    possible_paths = [
        'outputs/models/glaucoma_model.pth',
        'glaucoma_training/outputs/models/glaucoma_model.pth',
        'D:/AMD-project/outputs/models/glaucoma_model.pth'
    ]
    
    for path in possible_paths:
        try:
            model = GlaucomaClassifier(num_classes=2)
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            return model, checkpoint.get('val_acc', checkpoint.get('test_acc', None))
        except:
            continue
    
    return None, None

def predict_amd(image):
    """Predict eye disease from fundus image"""
    if amd_model is None:
        return None, None, None
    
    try:
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = amd_model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Get all probabilities
        all_probs = probs.cpu().numpy()[0]
        
        # Get prediction
        predicted_class = AMD_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        # Create results dictionary
        results = {AMD_CLASSES[i]: float(all_probs[i]) for i in range(len(AMD_CLASSES))}
        
        return predicted_class, confidence_score, results
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None

def predict_glaucoma(image):
    """Predict glaucoma from fundus image"""
    if glaucoma_model is None:
        return None, None, None
    
    try:
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = glaucoma_model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Get all probabilities
        all_probs = probs.cpu().numpy()[0]
        
        # Get prediction
        predicted_class = GLAUCOMA_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        # Create results dictionary
        results = {GLAUCOMA_CLASSES[i]: float(all_probs[i]) for i in range(len(GLAUCOMA_CLASSES))}
        
        return predicted_class, confidence_score, results
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None

def get_chat_response(question, diagnosis, confidence, all_results):
    """Generate chat response using LLM or rule-based system"""
    try:
        # Try using OpenAI API if available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Build context about the results
            context = f"""Patient Analysis Results:
- Primary Diagnosis: {diagnosis}
- Confidence: {confidence:.2f}%
- All Probabilities: {', '.join([f'{k}: {v*100:.2f}%' for k, v in all_results.items()])}

You are a helpful medical AI assistant. Answer questions about these eye disease detection results.
Provide clear, informative, but cautious responses. Always remind users to consult healthcare professionals.
Focus on: explaining the diagnosis, what it means, general information about the disease, prevention, and next steps."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": question}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            # Fallback to rule-based responses
            return get_rule_based_response(question, diagnosis, confidence)
    except Exception as e:
        return get_rule_based_response(question, diagnosis, confidence)

def get_rule_based_response(question, diagnosis, confidence):
    """Provide rule-based responses when LLM is not available"""
    question_lower = question.lower()
    
    # Disease information dictionary (needed for all responses)
    disease_info = {
        'AMD': {
            'full_name': 'Age-related Macular Degeneration',
            'description': "AMD affects the central part of the retina (macula), which is responsible for sharp, central vision. This condition typically develops in people over 50 and is one of the leading causes of vision loss in older adults. There are two types: dry AMD (more common, slower progression) and wet AMD (less common but more severe).",
            'symptoms': "Central vision becomes blurry or distorted, difficulty reading or recognizing faces, straight lines appear wavy, dark or empty areas in central vision.",
            'causes': "Aging is the primary factor, but genetics, smoking, high blood pressure, and poor diet also increase risk."
        },
        'Diabetic Retinopathy': {
            'full_name': 'Diabetic Retinopathy',
            'description': "This is a diabetes complication that damages the blood vessels in the retina. High blood sugar levels over time weaken and cause changes in the retinal blood vessels, which can leak fluid or bleed. If untreated, it can lead to severe vision loss or blindness.",
            'symptoms': "Early stages may have no symptoms. Later: blurred vision, floaters, dark areas in vision, difficulty seeing colors, vision loss.",
            'causes': "Prolonged high blood sugar levels from poorly controlled diabetes. Both Type 1 and Type 2 diabetes can cause this condition."
        },
        'Cataract': {
            'full_name': 'Cataract',
            'description': "A cataract is a clouding of the eye's natural lens, which lies behind the iris and pupil. It's like looking through a foggy or dusty window. Cataracts are very common in older adults and develop slowly over time. Most people develop cataracts in both eyes, though one may be worse than the other.",
            'symptoms': "Cloudy or blurry vision, faded colors, glare and halos around lights, poor night vision, double vision, frequent prescription changes.",
            'causes': "Aging is the most common cause. Other factors: diabetes, smoking, excessive alcohol, prolonged sun exposure, eye injury, certain medications."
        },
        'Glaucoma': {
            'full_name': 'Glaucoma',
            'description': "Glaucoma is a group of eye diseases that damage the optic nerve, usually due to increased pressure in the eye. The optic nerve carries visual information from the eye to the brain, and damage to it results in vision loss. It's often called the 'silent thief of sight' because early stages have no symptoms.",
            'symptoms': "Open-angle glaucoma: gradual peripheral vision loss, tunnel vision (advanced). Acute angle-closure: severe eye pain, nausea, blurred vision, halos around lights - this is an emergency!",
            'causes': "Increased intraocular pressure is the main risk factor, but glaucoma can occur with normal pressure. Age, family history, and certain medical conditions increase risk."
        },
        'Normal': {
            'full_name': 'Normal/Healthy Eye',
            'description': "Your retinal image shows no significant signs of the major eye diseases we screen for (AMD, Diabetic Retinopathy, Cataract, or Glaucoma). This is a positive result indicating healthy retinal structures.",
            'symptoms': "No abnormal symptoms detected.",
            'causes': "Not applicable - this is a healthy condition."
        }
    }
    
    current_disease_info = disease_info.get(diagnosis, {})
    
    # Priority handlers for specific requests
    
    # User is repeating/clarifying/frustrated (handle first!)
    if any(phrase in question_lower for phrase in ["i said", "i asked", "i'm asking", "im asking", "just tell me", "specifically", "be specific", "answer my question"]):
        # Check if it's about food
        if any(word in question_lower for word in ['food', 'eat', 'diet', 'nutrition']):
            # Redirect to food-specific response below
            question_lower = "what food should i eat"
        elif any(word in question_lower for word in ['treatment', 'cure', 'treat']):
            question_lower = "how is this treated"
    
    # Medication/supplement questions (handle early)
    if any(phrase in question_lower for phrase in ['tablet', 'tablets', 'medicine', 'medication', 'supplement', 'vitamins', 'pills', 'drugs', 'what to take', 'should i take', 'eye drops']):
        medication_info = {
            'AMD': """**Supplements & Medications for AMD:**

**AREDS2 Formula (Most Important!):**
- **Vitamin C:** 500 mg
- **Vitamin E:** 400 IU
- **Lutein:** 10 mg
- **Zeaxanthin:** 2 mg
- **Zinc:** 80 mg (as zinc oxide)
- **Copper:** 2 mg (to prevent zinc-induced copper deficiency)

**Who Should Take AREDS2:**
- People with intermediate AMD
- Advanced AMD in one eye
- **NOT for early AMD or prevention in healthy eyes**

**Available as:**
- PreserVision AREDS 2
- Bausch + Lomb OCUVITE
- Other brands with AREDS2 formula

**For Wet AMD - Injections (by doctor only):**
- Anti-VEGF injections (Lucentis, Eylea, Avastin)
- Given every 4-8 weeks
- Stop abnormal blood vessel growth

**⚠️ Important:**
- Don't take random eye supplements - use AREDS2 formula
- Consult ophthalmologist before starting
- High doses can interact with medications
- Not all supplements are proven effective

**Don't smoke!** Negates supplement benefits.""",

            'Diabetic Retinopathy': """**Medications for Diabetic Retinopathy:**

**Primary Treatment - Blood Sugar Control:**
- **Metformin** (if prescribed for diabetes)
- **Insulin** (if prescribed - NEVER skip doses!)
- Blood pressure medications (if prescribed)
- Cholesterol medications (statins if needed)

**Key Point:** Blood sugar control IS the primary medication treatment!

**Eye-Specific Treatments (by doctor only):**
- **Anti-VEGF Injections:** Lucentis, Eylea, Avastin
  - Reduces swelling and abnormal blood vessels
  - Given every 4-8 weeks
- **Steroid Injections:** For macular edema
  - Ozurdex, Iluvien implants

**Supplements (May Help):**
- **Omega-3 Fish Oil:** 1000-2000 mg/day
- **Vitamin D:** If deficient
- **Alpha-lipoic acid:** May help nerve health (ask doctor)

**NOT Recommended:**
- Random "eye health" supplements
- Unproven herbal remedies

**⚠️ CRITICAL:**
- Focus on diabetes medications first!
- Never replace prescribed meds with supplements
- Keep HbA1c below 7%
- Consult endocrinologist for diabetes meds
- See ophthalmologist for eye injections

**Your diabetes medication regimen is MORE important than any eye supplement!**""",

            'Cataract': """**Medications & Supplements for Cataract:**

**Truth About Cataracts:**
**There are NO medications or supplements that can remove or reverse cataracts.**
**Surgery is the ONLY treatment that actually removes cataracts.**

**Supplements (May Slow Progression - Not Proven):**
- **Vitamin C:** 500 mg/day (antioxidant)
- **Vitamin E:** 400 IU/day
- **Lutein & Zeaxanthin:** 10-20 mg combined
- **Omega-3 Fish Oil:** 1000 mg/day

**Evidence:** Limited - may help slow early cataracts but won't remove them

**Before Surgery (Symptom Management):**
- **Stronger glasses prescription**
- **Anti-glare coatings on lenses**
- **Dilating eye drops** (rarely used, temporary relief)

**⚠️ Beware of Scams:**
- "Miracle eye drops that dissolve cataracts" - DON'T EXIST
- "Natural cataract cure" - NO such thing
- Lanosterol drops - still experimental, not proven

**When Cataracts Interfere with Life:**
- **Cataract surgery** is the answer
- Very safe (>95% success)
- Outpatient procedure
- Vision improvement is dramatic

**Bottom Line:**
- Supplements MAY slow early progression (weak evidence)
- NO medication removes cataracts
- Surgery is highly effective when needed
- Don't waste money on "cataract cure" products

**Consult ophthalmologist about:**
- Whether surgery is needed now
- If supplements are worth trying (usually not necessary)""",

            'Glaucoma': """**Medications for Glaucoma:**

**Eye Drops (Primary Treatment - Daily Use!):**

**1. Prostaglandin Analogs (Most Common, Once Daily):**
- Latanoprost (Xalatan)
- Travoprost (Travatan)
- Bimatoprost (Lumigan)
- Lower eye pressure 25-30%
- Side effect: longer lashes, eye color change

**2. Beta-Blockers (1-2 times daily):**
- Timolol (Timoptic)
- Betaxolol
- Lower fluid production
- Caution with asthma, heart conditions

**3. Alpha Agonists (2-3 times daily):**
- Brimonidine (Alphagan)
- Apraclonidine
- Reduce fluid production

**4. Carbonic Anhydrase Inhibitors:**
- Dorzolamide (Trusopt)
- Brinzolamide (Azopt)
- Oral: Acetazolamide (Diamox)
- Reduce fluid production

**5. Combination Drops:**
- Cosopt (dorzolamide + timolol)
- Combigan (brimonidine + timolol)
- Convenience of fewer drops

**Oral Medications (If drops insufficient):**
- **Acetazolamide (Diamox)** - for acute cases
- Side effects: frequent urination, tingling

**Supplements (Supporting Role Only):**
- **Ginkgo biloba** - may improve blood flow (weak evidence)
- **Vitamin B complex** - for nerve health
- **NOT a substitute for prescription drops!**

**⚠️ CRITICAL RULES:**
- **NEVER skip your eye drops** - even one day damages optic nerve
- Use exactly as prescribed
- Wait 5 minutes between different eye drops
- Store properly
- Don't touch dropper to eye
- Refill before running out

**What Drops DON'T Do:**
- Don't restore lost vision (only prevent further loss)
- Must use for life (usually)

**If drops not enough:**
- Laser treatment (SLT, LPI)
- Surgery (trabeculectomy, tube shunts, MIGS)

**Consult ophthalmologist to:**
- Get prescription eye drops
- Determine best medication for you
- Monitor eye pressure regularly""",

            'Normal': """**Supplements for Healthy Eyes:**

**Good News: You don't need special medications!**

**If You Want to Take Eye Supplements (Optional):**

**General Eye Health Formula:**
- **Lutein:** 10 mg
- **Zeaxanthin:** 2 mg
- **Omega-3s:** 1000 mg (fish oil)
- **Vitamin C:** 500 mg
- **Vitamin E:** 400 IU
- **Zinc:** 15-25 mg

**Available as:**
- Bausch + Lomb Ocuvite
- PreserVision (non-AREDS formula for healthy eyes)
- Nature Made Vision supplements

**Important:**
- **You DON'T need these if you eat a balanced diet!**
- Leafy greens, fish, colorful vegetables provide these naturally
- Supplements are "insurance" not necessity

**What You DON'T Need:**
- AREDS2 formula (that's for AMD, not healthy eyes)
- Expensive "miracle" eye formulas
- Multiple different eye supplements

**Better Approach:**
- Eat healthy diet rich in eye nutrients
- Protect eyes from UV (sunglasses)
- Regular eye exams
- Don't smoke

**Only Consider Supplements If:**
- Diet is poor
- You have risk factors (family history, age >50)
- Your doctor recommends them

**Consult ophthalmologist if you're considering supplements.**

Keep up the healthy habits - prevention is best medicine!"""
        }
        return medication_info.get(diagnosis, f"For {diagnosis}, consult your ophthalmologist about appropriate medications or supplements. Treatment varies by condition severity and individual needs.")
    
    # Foods to AVOID questions (more specific, handle first)
    if any(phrase in question_lower for phrase in ['food to avoid', 'foods to avoid', 'what to avoid', 'avoid eating', 'should not eat', "shouldn't eat", 'stay away from', 'bad foods', 'foods i cant', "can't eat", 'stop eating']):
        foods_to_avoid = {
            'AMD': """**Foods to AVOID with AMD:**

**❌ Processed & Fried Foods:**
- French fries, fried chicken, donuts
- Increase inflammation and oxidative stress
- Linked to AMD progression

**❌ Foods High in Saturated Fats:**
- Fatty red meat, butter, full-fat dairy
- Can worsen AMD

**❌ Refined Carbohydrates:**
- White bread, white rice, pastries
- Cookies, cakes, white pasta
- High glycemic index = higher AMD risk

**❌ Trans Fats:**
- Margarine, shortening
- Commercial baked goods
- Packaged snacks with "partially hydrogenated oils"

**❌ Excessive Sugar:**
- Sodas, candy, sweetened beverages
- Can accelerate eye aging

**❌ High-Sodium Foods:**
- Processed meats (bacon, sausage, deli meats)
- Canned soups, frozen dinners
- Salty snacks

**❌ Alcohol (in excess):**
- Limit to 1 drink/day for women, 2 for men
- Excessive alcohol increases AMD risk

**Instead Choose:**
- Leafy greens, fish, whole grains, nuts, colorful vegetables""",

            'Diabetic Retinopathy': """**Foods to AVOID with Diabetic Retinopathy:**

**❌ Sugary Foods & Drinks (CRITICAL!):**
- Sodas, energy drinks, sweetened tea/coffee
- Candy, cookies, cakes, pastries
- Ice cream, donuts
- Fruit juices (even "natural" ones)
- Spike blood sugar rapidly = damage blood vessels

**❌ White/Refined Carbs:**
- White bread, white rice, white pasta
- Instant oatmeal, sugary cereals
- Crackers, pretzels
- Convert to sugar quickly

**❌ Fried Foods:**
- French fries, fried chicken, tempura
- Increase inflammation
- Worsen diabetic complications

**❌ High-Saturated Fat Foods:**
- Fatty red meat, bacon, sausage
- Full-fat dairy, butter
- Can worsen insulin resistance

**❌ Trans Fats:**
- Margarine, shortening
- Packaged baked goods, microwave popcorn
- Foods with "partially hydrogenated oils"

**❌ High-Sodium Foods:**
- Processed meats, canned soups
- Fast food, frozen dinners
- High blood pressure worsens retinopathy

**❌ Alcohol (especially if blood sugar unstable):**
- Can cause dangerous blood sugar swings
- Empty calories

**Critical:** These foods spike blood sugar → damage retinal blood vessels → worsen retinopathy!

**Stick to:** Non-starchy vegetables, lean proteins, whole grains in moderation, low-GI foods""",

            'Cataract': """**Foods to AVOID for Cataract Prevention:**

**❌ Excessive Sugar:**
- Sodas, candy, sweetened beverages
- Desserts, pastries, cookies
- Can accelerate cataract formation
- Sugar causes lens protein changes

**❌ High-Sodium Foods:**
- Processed meats (bacon, sausage, deli meats)
- Canned soups, frozen dinners
- Salty snacks (chips, pretzels)
- May contribute to cataract risk

**❌ Fried Foods:**
- French fries, fried chicken, donuts
- Increase oxidative stress
- Damage lens proteins

**❌ Processed Foods:**
- Fast food, packaged meals
- Foods with many preservatives
- Lack nutrients, high in unhealthy fats

**❌ Excessive Alcohol:**
- More than 2 drinks/day increases cataract risk
- Can deplete vitamins needed for eye health
- Dehydrates the body and eyes

**❌ Trans Fats:**
- Margarine, shortening
- Commercial baked goods
- Promote inflammation

**❌ Refined Carbs:**
- White bread, white rice
- Sugary cereals
- Better to choose whole grains

**Moderation is Key:**
- Occasional treats are OK
- Focus on mostly whole, unprocessed foods
- Stay hydrated

**Choose Instead:** Colorful fruits/vegetables, whole grains, lean proteins, water""",

            'Glaucoma': """**Foods to AVOID with Glaucoma:**

**❌ Excessive Caffeine:**
- Limit coffee to 1-2 cups/day
- Energy drinks, excessive tea
- Can temporarily increase eye pressure
- Effects last 90 minutes after consumption

**❌ Trans Fats:**
- Margarine, shortening, fried foods
- Packaged baked goods
- May increase glaucoma risk

**❌ Large Amounts of Liquid at Once:**
- Don't chug a quart of water
- Drink throughout the day instead
- Sudden large intake can spike eye pressure

**❌ High-Sodium Foods:**
- Processed meats, canned soups
- Fast food, frozen dinners
- High blood pressure worsens glaucoma

**❌ Refined Carbs & Sugar:**
- Can affect blood sugar and circulation
- White bread, sugary snacks
- May impact optic nerve health

**❌ Excessive Alcohol:**
- Can temporarily lower then increase pressure
- Limit to moderate amounts
- Avoid binge drinking

**Important Notes:**
- Diet is supportive, NOT primary treatment
- Don't skip glaucoma medications thinking diet alone will help
- Some foods (like caffeine) have temporary effects

**Safe Choices:** Leafy greens, omega-3 fish, berries, whole grains, green tea (moderate)""",

            'Normal': """**Foods to Limit for Eye Health:**

**Minimize These:**

**❌ Processed & Fried Foods:**
- Can increase oxidative stress over time
- Choose baked or grilled instead

**❌ Excessive Sugar:**
- Linked to higher risk of eye diseases
- Enjoy sweets occasionally, not daily

**❌ Trans Fats:**
- Promote inflammation
- Check labels for "partially hydrogenated oils"

**❌ High Sodium:**
- Can affect blood pressure
- Which impacts eye health

**❌ Excessive Alcohol:**
- Moderate consumption (if any) is fine
- 1 drink/day for women, 2 for men max

**Good News:**
Since your eyes are healthy, you don't need strict restrictions! Just maintain a balanced diet with mostly whole foods.

**Focus on moderation and variety!**"""
        }
        return foods_to_avoid.get(diagnosis, f"To maintain eye health with {diagnosis}, limit processed foods, excessive sugar, trans fats, and fried foods. Focus on whole, nutritious foods instead.")
    
    # Specific food/diet/nutrition questions (handle before general prevention)
    if any(phrase in question_lower for phrase in ['what food', 'which food', 'foods to eat', 'what to eat', 'diet for', 'best foods', 'food recommendations', 'nutrition', 'eating habits', 'what should i eat', 'foods help']):
        food_recommendations = {
            'AMD': """**Best Foods for AMD Prevention:**

**Leafy Greens (Most Important!):**
- Kale, spinach, collard greens
- Swiss chard, arugula
- Rich in lutein & zeaxanthin (protect macula)
- Aim for 1-2 servings daily

**Fish (Omega-3s):**
- Salmon, tuna, sardines, mackerel
- 2-3 servings per week
- Reduces inflammation

**Colorful Fruits & Vegetables:**
- Oranges, berries, carrots, bell peppers
- Tomatoes, sweet potatoes
- Rich in vitamins C and E

**Nuts & Seeds:**
- Almonds, walnuts, sunflower seeds
- Good source of vitamin E

**Whole Grains:**
- Oatmeal, quinoa, brown rice
- Better than refined carbs

**What to Avoid:**
- Processed foods high in refined carbs
- Trans fats and saturated fats
- Excessive red meat

**Sample Daily Plate:**
- Breakfast: Oatmeal with berries and walnuts
- Lunch: Spinach salad with salmon
- Dinner: Grilled fish with sweet potato and steamed kale
- Snacks: Oranges, carrots, almonds""",

            'Diabetic Retinopathy': """**Best Foods for Diabetic Retinopathy Prevention:**

**The key is controlling blood sugar while nourishing your eyes!**

**Low Glycemic Index Foods (Essential!):**
- Steel-cut oatmeal (not instant)
- Quinoa, barley, bulgur
- Legumes: lentils, chickpeas, beans
- Keep blood sugar stable

**Leafy Greens & Non-Starchy Vegetables:**
- Spinach, kale, broccoli, cauliflower
- Cucumbers, tomatoes, peppers
- Won't spike blood sugar
- Rich in eye-protective nutrients

**Fish High in Omega-3s:**
- Salmon, mackerel, sardines
- 2-3 times per week
- Anti-inflammatory properties

**Berries (Low Sugar Fruits):**
- Blueberries, strawberries, blackberries
- Rich in antioxidants
- Lower glycemic impact than other fruits

**Nuts (in moderation):**
- Almonds, walnuts (1 handful/day)
- Good fats, don't spike blood sugar

**What to AVOID (Critical!):**
- White bread, white rice, pastries
- Sugary drinks, sodas, fruit juices
- Candy, cookies, processed sweets
- Fried foods
- Excessive carbohydrates

**Meal Planning Tips:**
- Fill half your plate with non-starchy vegetables
- Quarter with lean protein (fish, chicken)
- Quarter with whole grains or legumes
- Eat at regular times
- Pair carbs with protein/fat to slow absorption

**Sample Diabetic-Friendly Day:**
- Breakfast: Steel-cut oats with berries and almonds
- Snack: Carrot sticks with hummus
- Lunch: Grilled salmon on spinach salad
- Snack: Small apple with peanut butter
- Dinner: Grilled chicken, quinoa, roasted broccoli""",

            'Cataract': """**Best Foods for Cataract Prevention:**

**Antioxidant-Rich Foods (Fight oxidative stress!):**

**Colorful Fruits:**
- Berries: blueberries, strawberries, raspberries
- Citrus: oranges, grapefruit, lemons (vitamin C)
- Kiwi, papaya, mango

**Vegetables:**
- Spinach, kale, broccoli (lutein & zeaxanthin)
- Carrots, sweet potatoes (beta-carotene)
- Bell peppers (vitamin C)
- Tomatoes (lycopene)

**Vitamin E Sources:**
- Almonds, sunflower seeds
- Avocados
- Olive oil

**Omega-3 Rich Fish:**
- Salmon, trout, sardines
- May slow cataract progression

**Whole Grains:**
- Brown rice, quinoa, whole wheat
- Better than refined grains

**What to Limit:**
- Excessive alcohol
- High-sodium foods
- Processed foods

**Bonus Tips:**
- Drink plenty of water (hydration helps)
- Green tea (antioxidants)
- Limit sugar (can accelerate cataract formation)

**Sample Day:**
- Breakfast: Greek yogurt with berries and almonds
- Snack: Orange
- Lunch: Salmon with spinach and tomatoes
- Snack: Carrot sticks
- Dinner: Grilled chicken, sweet potato, steamed broccoli""",

            'Glaucoma': """**Best Foods for Glaucoma Management:**

**Dark Leafy Greens (Most Beneficial!):**
- Kale, collard greens, spinach
- High in nitrates (may lower eye pressure)
- Rich in antioxidants

**Foods High in Vitamins A, C, E:**
- Carrots, sweet potatoes (vitamin A)
- Citrus fruits, bell peppers (vitamin C)
- Almonds, sunflower seeds (vitamin E)

**Omega-3 Rich Fish:**
- Salmon, tuna, sardines
- May help with eye pressure

**Berries & Dark-Colored Fruits:**
- Blueberries, blackberries, cranberries
- Rich in antioxidants

**Foods with Zinc:**
- Lean beef, poultry
- Legumes, nuts
- Whole grains

**Tea (Especially Green Tea):**
- May help lower eye pressure
- Rich in antioxidants

**What to Avoid:**
- Excessive caffeine (may temporarily increase pressure)
- Trans fats
- Very high fluid intake in short period

**Important Note:**
Diet supports but doesn't replace medical treatment! Medications are still primary treatment.

**Sample Day:**
- Breakfast: Oatmeal with berries and walnuts
- Lunch: Spinach salad with grilled salmon
- Snack: Carrot sticks, handful of almonds
- Dinner: Baked chicken with kale and sweet potato
- Drink: Green tea throughout the day""",

            'Normal': """**Foods for Maintaining Healthy Eyes:**

**Keep your eyes healthy with these nutrient-rich foods:**

**Leafy Greens:**
- Spinach, kale, collard greens
- Protect against eye diseases

**Colorful Vegetables:**
- Carrots, bell peppers, sweet potatoes
- Rich in beta-carotene and vitamins

**Fatty Fish:**
- Salmon, tuna, sardines
- Omega-3s for retinal health

**Citrus Fruits & Berries:**
- Oranges, berries, kiwi
- Antioxidants and vitamin C

**Nuts & Seeds:**
- Almonds, walnuts, chia seeds
- Vitamin E and healthy fats

**Eggs:**
- Lutein, zeaxanthin, zinc
- Complete eye nutrition

**Balanced Diet Principles:**
- Variety of colors on your plate
- Whole foods over processed
- Stay hydrated
- Moderate portions

Keep up the good work maintaining your eye health!"""
        }
        return food_recommendations.get(diagnosis, f"For optimal eye health with {diagnosis}, focus on leafy greens, colorful fruits and vegetables, fatty fish rich in omega-3s, and nuts. Consult a nutritionist for a personalized diet plan.")
    
    # More flexible question matching with direct answers
    
    # Questions about what/explain the disease
    if any(word in question_lower for word in ['what is', 'explain', 'tell me about', 'describe']):
        return f"""**{current_disease_info.get('full_name', diagnosis)}**

{current_disease_info.get('description', 'This is an eye condition detected in your retinal image.')}

**Common Symptoms:** {current_disease_info.get('symptoms', 'Varies by condition')}

**What Causes It:** {current_disease_info.get('causes', 'Multiple factors can contribute')}

Your current diagnosis shows **{diagnosis}** with **{confidence:.1f}% confidence**.

⚕️ *Important: This is an AI screening tool. Please consult an ophthalmologist for comprehensive evaluation and personalized medical advice.*"""
    
    # Questions about treatment
    elif any(word in question_lower for word in ['treatment', 'cure', 'treat', 'fix', 'heal', 'medicine', 'surgery']):
        treatments = {
            'AMD': """**Treatment Options for AMD:**

**Dry AMD:**
- AREDS2 vitamin supplements (antioxidants, zinc, copper)
- Lifestyle modifications (quit smoking, healthy diet)
- Low vision aids for advanced cases

**Wet AMD:**
- Anti-VEGF injections (most effective - blocks abnormal blood vessel growth)
- Photodynamic therapy
- Laser therapy

**Lifestyle Changes:**
- Eat leafy greens, fish high in omega-3s
- Protect eyes from UV light
- Monitor vision with Amsler grid
- Regular eye exams""",
            
            'Diabetic Retinopathy': """**Treatment Options for Diabetic Retinopathy:**

**Primary Treatment:**
- **Blood sugar control** - This is THE most important factor
- Maintain HbA1c below 7%
- Control blood pressure and cholesterol

**Medical Interventions:**
- Anti-VEGF injections (reduces swelling and abnormal vessels)
- Laser photocoagulation (seals leaking vessels)
- Vitrectomy (surgery for advanced cases with bleeding)
- Steroid injections (for swelling)

**Prevention of Progression:**
- Regular comprehensive eye exams (every 3-12 months)
- Strict diabetes management
- Healthy lifestyle (diet, exercise)""",
            
            'Cataract': """**Treatment Options for Cataract:**

**Surgery (Only Definitive Treatment):**
- **Phacoemulsification** - Most common, ultrasound breaks up cloudy lens
- Lens is removed and replaced with artificial intraocular lens (IOL)
- Outpatient procedure, usually 15-30 minutes
- Very high success rate (>95%)
- Vision improvement is usually immediate to a few days

**Before Surgery (if mild):**
- Stronger eyeglasses or contacts
- Brighter lighting for reading
- Anti-glare sunglasses
- Magnifying lenses

**When to Have Surgery:**
- When vision loss interferes with daily activities
- Difficulty driving, reading, or working
- Failed vision test for driver's license

Surgery is very safe and effective - don't let fear delay treatment!""",
            
            'Glaucoma': """**Treatment Options for Glaucoma:**

**Goal:** Lower eye pressure to prevent optic nerve damage

**Medications (First-Line):**
- Eye drops (prostaglandins, beta-blockers, alpha agonists)
- Must be used daily, exactly as prescribed
- Multiple types often combined

**Laser Treatment:**
- Trabeculoplasty (improves drainage)
- Iridotomy (for angle-closure type)
- Cyclophotocoagulation (reduces fluid production)

**Surgery (if medications insufficient):**
- Trabeculectomy (creates new drainage)
- Drainage implants
- Minimally invasive glaucoma surgery (MIGS)

**Critical Points:**
- Treatment doesn't restore lost vision, only prevents further loss
- Lifelong treatment usually required
- Regular monitoring essential
- Never skip medications""",
            
            'Normal': """**Maintaining Healthy Eyes:**

Since your eyes appear healthy, focus on prevention:

- **Regular Eye Exams:** Every 1-2 years (annually if over 60)
- **Healthy Diet:** Leafy greens, fish, colorful fruits/vegetables
- **Protect Eyes:** UV-blocking sunglasses, safety glasses when needed
- **Manage Health:** Control blood sugar, blood pressure, cholesterol
- **Don't Smoke:** Smoking doubles risk of many eye diseases
- **Exercise Regularly:** Improves blood flow to eyes
- **Good Lighting:** Reduce eye strain when reading/working
- **20-20-20 Rule:** Every 20 min, look 20 feet away for 20 seconds

Keep up the good work! Prevention is the best medicine."""
        }
        
        return f"{treatments.get(diagnosis, 'Treatment varies by condition and severity.')}\n\n⚠️ **Always consult a qualified ophthalmologist before starting any treatment. This information is educational only.**"
    
    # Questions about prevention
    elif any(word in question_lower for word in ['prevent', 'avoid', 'stop', 'reduce risk', 'protect']):
        prevention = {
            'AMD': """**Preventing AMD Progression:**

- **Stop Smoking** - Single most important action! Smokers have 2-3x higher risk
- **Healthy Diet:** Leafy greens (kale, spinach), fish (salmon, tuna), nuts, citrus fruits
- **AREDS2 Supplements** - Vitamins C, E, zinc, copper, lutein, zeaxanthin (ask doctor)
- **UV Protection** - Wear sunglasses outdoors
- **Manage Blood Pressure** - Keep it controlled
- **Exercise Regularly** - Improves circulation
- **Maintain Healthy Weight**
- **Monitor Vision** - Use Amsler grid weekly, report changes immediately""",
            
            'Diabetic Retinopathy': """**Preventing Diabetic Retinopathy:**

**Most Critical:**
- **Control Blood Sugar** - Keep HbA1c < 7% (work with endocrinologist)
- **Blood Pressure** - Keep below 130/80 mmHg
- **Cholesterol** - Maintain healthy lipid levels

**Lifestyle:**
- Follow diabetic diet plan
- Exercise 30+ minutes daily
- Take medications as prescribed
- Don't skip insulin doses
- Monitor blood sugar regularly

**Eye Care:**
- Dilated eye exam at least annually (more if DR detected)
- Report vision changes immediately
- Don't wait for symptoms - damage can occur without symptoms

**Other:**
- Maintain healthy weight
- Don't smoke
- Limit alcohol
- Manage stress""",
            
            'Cataract': """**Preventing or Slowing Cataract Development:**

While cataracts are often age-related and inevitable, you can slow progression:

- **Protect from UV** - Wear UV-blocking sunglasses and wide-brimmed hat
- **Don't Smoke** - Significantly increases cataract risk
- **Limit Alcohol** - Excessive drinking increases risk
- **Manage Diabetes** - High blood sugar accelerates cataract formation
- **Healthy Diet:** Antioxidant-rich foods (berries, citrus, leafy greens)
- **Avoid Eye Injury** - Wear protective eyewear during sports/work
- **Regular Eye Exams** - Early detection allows better planning
- **Medication Review** - Some medications (steroids) can cause cataracts

**Note:** Once cataracts develop, only surgery can remove them. These measures help slow progression.""",
            
            'Glaucoma': """**Preventing Glaucoma and Vision Loss:**

**You Cannot Prevent Glaucoma, But You Can Prevent Blindness:**

- **Regular Eye Exams** - Critical! Every 1-2 years, annually after 40
- **Know Your Family History** - Glaucoma is hereditary
- **Use Medications as Prescribed** - If diagnosed, never skip eye drops
- **Protect Eyes** - Wear protective eyewear (injury can cause glaucoma)
- **Exercise Regularly** - May help lower eye pressure
- **Maintain Healthy Weight**
- **Avoid High Eye Pressure Activities:**
  - Don't hold breath or strain excessively
  - Elevate head while sleeping
  - Limit caffeine
  - Stay hydrated

**High-Risk Groups Should Be Extra Vigilant:**
- Age > 60
- African American, Asian, or Hispanic
- Family history of glaucoma
- High eye pressure
- Diabetes, high blood pressure""",
            
            'Normal': """**Keeping Your Eyes Healthy:**

Great news - your eyes are healthy! Here's how to keep them that way:

**Essential Habits:**
- **Regular Eye Exams:** Every 1-2 years (annually if over 60 or at risk)
- **Wear Sunglasses:** 100% UV protection, even on cloudy days
- **Healthy Lifestyle:** Don't smoke, limit alcohol, exercise, manage stress
- **Protective Eyewear:** Safety glasses for sports, work, yard work

**Nutrition for Eye Health:**
- Leafy greens (lutein, zeaxanthin)
- Fish with omega-3s (salmon, tuna)
- Colorful fruits/vegetables (antioxidants)
- Nuts and seeds (vitamin E)
- Citrus fruits (vitamin C)

**Digital Eye Strain Prevention:**
- 20-20-20 rule
- Proper screen distance (arm's length)
- Reduce glare
- Blink frequently
- Use artificial tears if needed

**Manage Health Conditions:**
- Keep blood sugar controlled if diabetic
- Maintain healthy blood pressure
- Monitor cholesterol levels

Continue these habits to maintain your healthy vision!"""
        }
        
        return f"{prevention.get(diagnosis, 'Prevention strategies vary by condition.')}\n\n⚕️ *Consult your eye care professional for personalized prevention strategies.*"
    
    # Questions about confidence/accuracy
    elif any(word in question_lower for word in ['confidence', 'accurate', 'sure', 'reliable', 'trust', 'correct']):
        if confidence > 90:
            confidence_level = "very high confidence"
            interpretation = "The model is quite certain about this diagnosis."
        elif confidence > 70:
            confidence_level = "high confidence"
            interpretation = "The model shows strong confidence in this diagnosis."
        elif confidence > 50:
            confidence_level = "moderate confidence"
            interpretation = "The model shows reasonable confidence, but professional verification is important."
        else:
            confidence_level = "lower confidence"
            interpretation = "The model is less certain. Professional examination is especially important."
        
        return f"""**About Your Result:**

The model detected **{diagnosis}** with **{confidence:.1f}% confidence**, which represents {confidence_level}. {interpretation}

**Understanding the AI Model:**
- **Architecture:** ResNet50 deep learning neural network
- **Training:** Trained on thousands of retinal fundus images
- **Process:** Analyzes patterns in blood vessels, retinal structures, and abnormalities
- **Multi-Disease:** Can detect AMD, Diabetic Retinopathy, Cataract, Glaucoma, and Normal eyes

**Important Limitations:**

✓ **What It Does Well:**
- Quick screening of large numbers of images
- Consistent analysis without fatigue
- Detection of subtle patterns humans might miss

✗ **What It Cannot Do:**
- Replace comprehensive medical examination
- Consider patient history and symptoms
- Detect all eye conditions (only those it's trained on)
- Account for image quality issues
- Provide definitive diagnosis

**Accuracy Factors:**
- Image quality affects results
- Some conditions are harder to detect than others
- False positives and false negatives can occur
- Early-stage diseases may be missed

**Your Next Step:**
Even with {confidence_level}, you should schedule an appointment with an ophthalmologist for:
- Complete eye examination
- Medical history review
- Additional diagnostic tests if needed
- Professional diagnosis and treatment plan

Think of this AI as a screening tool that flags potential issues for professional review, not as a replacement for an eye doctor."""
    
    # Questions about next steps/actions
    elif any(word in question_lower for word in ['next', 'should', 'do', 'action', 'now what', 'what now']):
        steps = {
            'AMD': """**Your Action Plan for AMD:**

**Immediate Steps (Next Few Days):**
1. 📞 **Call an Ophthalmologist** - Request appointment specifically for AMD evaluation
2. 📋 **Document Your Symptoms** - Note any vision changes, when they started
3. 📸 **Keep This Result** - Bring it to your appointment

**At Your Appointment, Expect:**
- Dilated eye examination
- OCT scan (optical coherence tomography) - shows retinal layers
- Fluorescein angiography (if wet AMD suspected)
- Visual acuity test
- Amsler grid test

**Ongoing Management:**
- Regular follow-up exams (frequency depends on severity)
- Monitor vision at home with Amsler grid
- Take prescribed supplements (AREDS2)
- Implement lifestyle changes

**Questions to Ask Your Doctor:**
- Is it dry or wet AMD?
- What stage am I in?
- How often should I be monitored?
- Should I take AREDS2 supplements?
- Are there any treatments available for me?
- What's my prognosis?

**Timeline:**
- Make appointment: Within 1-2 weeks
- Treatment (if needed): Varies by type and severity
- Follow-up: Every 3-6 months typically""",
            
            'Diabetic Retinopathy': """**Your Action Plan for Diabetic Retinopathy:**

**Urgent Actions (This Week):**
1. 📞 **Contact Ophthalmologist** - Mention diabetic retinopathy screening result
2. 📊 **Check Recent HbA1c** - Know your blood sugar control status
3. 👨‍⚕️ **Inform Your Endocrinologist/PCP** - Coordinate care between doctors
4. 📋 **Track Blood Sugar** - Start monitoring more closely if not already

**At Your Eye Appointment:**
- Comprehensive dilated exam
- Retinal photography
- OCT scan (checks for swelling/macular edema)
- Fluorescein angiography (maps blood vessel damage)

**Critical Coordination:**
- Your eye doctor and diabetes doctor should communicate
- Better diabetes control is PRIMARY treatment
- May need adjustment to diabetes medications

**Questions for Ophthalmologist:**
- What stage of DR do I have? (mild, moderate, severe, proliferative)
- Is there macular edema?
- Do I need treatment now or just monitoring?
- How often should I have eye exams?

**Questions for Diabetes Doctor:**
- Is my HbA1c optimal? (target < 7%)
- Should we adjust my diabetes medications?
- How can I improve blood sugar control?

**Prevention of Progression:**
- Strict blood sugar control (MOST IMPORTANT)
- Blood pressure management
- Regular monitoring

**Timeline:**
- Eye appointment: Within 1-2 weeks
- Diabetes review: Next available
- Follow-up eye exams: Every 3-12 months depending on severity""",
            
            'Cataract': """**Your Action Plan for Cataract:**

**Non-Urgent Steps (Plan Within 1-2 Months):**
1. 📞 **Schedule Eye Exam** - Regular ophthalmologist or optometrist
2. 📝 **Assess Impact** - How much is vision affecting your daily life?
3. 🚗 **Consider Your Activities** - Is it affecting driving, reading, work?

**Good News:** Cataracts are NOT an emergency and progress slowly!

**At Your Appointment:**
- Visual acuity test
- Slit lamp examination
- Dilated exam to see cataract and check overall eye health
- Discussion of surgery timing

**When to Consider Surgery:**
- Vision loss interferes with daily activities
- Failed driver's vision test
- Difficulty reading, watching TV, recognizing faces
- Glare causing problems with night driving

**Surgery Information:**
- Very common and safe (>95% success rate)
- Outpatient procedure (15-30 minutes)
- Usually one eye at a time, 2-4 weeks apart
- Quick recovery (days to weeks)
- Insurance typically covers if medically necessary

**Questions for Your Doctor:**
- How advanced is my cataract?
- Is surgery recommended now or should we wait?
- What type of intraocular lens would you recommend?
- What are the risks and benefits?
- What's the recovery like?

**If Not Ready for Surgery:**
- Stronger glasses prescription
- Better lighting for reading
- Anti-glare sunglasses for driving
- Regular monitoring

**Timeline:**
- Initial consult: Within 1-2 months
- Surgery (if needed): When you and doctor agree timing is right
- Often people wait until it significantly impacts life""",
            
            'Glaucoma': """**Your Action Plan for Glaucoma:**

**IMPORTANT - More Urgent Than Some Other Conditions:**

**Immediate Steps (This Week):**
1. 📞 **Schedule Ophthalmologist Appointment SOON** - Glaucoma can cause permanent vision loss
2. 📋 **Note Any Symptoms** - Peripheral vision loss, eye pain, headaches
3. 👪 **Check Family History** - Ask relatives about glaucoma

**Why Urgency Matters:**
- Glaucoma damage is PERMANENT and IRREVERSIBLE
- Early treatment prevents vision loss
- You can't feel high eye pressure - no symptoms until damage occurs
- Called "silent thief of sight" for a reason

**At Your Appointment, Expect:**
- **Tonometry** - Measures eye pressure (most important)
- **Ophthalmoscopy** - Examines optic nerve for damage
- **Perimetry** - Visual field test (checks peripheral vision)
- **Gonioscopy** - Examines drainage angle
- **OCT** - Measures optic nerve fiber layer thickness
- **Pachymetry** - Measures corneal thickness

**Possible Outcomes:**
- Confirmed glaucoma → Start treatment immediately
- Glaucoma suspect → Close monitoring
- False positive → Relief, but continue regular exams

**If Diagnosed, Treatment Starts Immediately:**
- Daily eye drops (MUST be used exactly as prescribed)
- Regular monitoring (every 3-6 months)
- Possible laser treatment or surgery if medications insufficient

**Critical Questions for Doctor:**
- What's my eye pressure?
- Is there optic nerve damage?
- What stage of glaucoma?
- What treatment do you recommend?
- How often do I need follow-up?
- What's my prognosis with treatment?

**Key Points:**
- Treatment doesn't restore vision, only preserves what you have
- Lifelong treatment usually required
- Never skip medications - one missed dose can affect progress

**Timeline:**
- Appointment: Within 1-2 weeks (SOONER if symptoms)
- Treatment: Begins immediately if diagnosed
- Follow-up: Every 3-6 months for life""",
            
            'Normal': """**Your Action Plan for Healthy Eyes:**

**Great News!** No immediate action required, but maintain good eye health:

**Continue Regular Check-ups:**
- **Under 40:** Every 2-3 years
- **40-54:** Every 2-4 years  
- **55-64:** Every 1-3 years
- **65+:** Every 1-2 years
- **High Risk:** Annually (diabetes, family history, high myopia)

**Maintain Healthy Habits:**
✓ Protect from UV (sunglasses)
✓ Eat eye-healthy foods
✓ Don't smoke
✓ Exercise regularly
✓ Manage health conditions
✓ Use protective eyewear when needed
✓ Give eyes breaks from screens

**When to Schedule Earlier:**
- Sudden vision changes
- Flashes of light or new floaters
- Eye pain
- Red or irritated eyes that don't improve
- Difficulty seeing at night
- Distorted vision

**Questions for Next Routine Exam:**
- Are there any early signs I should watch for?
- Am I at increased risk for any conditions?
- Is my eye pressure normal?
- Should I take any preventive measures?

**Stay Proactive:**
Even though your eyes are healthy now, regular monitoring helps catch problems early when they're most treatable.

**Timeline:**
- Next routine exam: Based on your age and risk factors
- Sooner if you notice any changes

Keep up the good work maintaining your eye health!"""
        }
        
        return steps.get(diagnosis, "Schedule an appointment with an ophthalmologist to discuss these results and determine the best course of action.")
    
    # Questions about symptoms
    elif any(word in question_lower for word in ['symptom', 'feel', 'notice', 'sign', 'experience']):
        symptoms = current_disease_info.get('symptoms', 'Symptoms vary by condition.')
        return f"""**Symptoms of {diagnosis}:**

{symptoms}

**Important Notes:**
- Early stages may have NO symptoms
- Symptoms can develop gradually
- You might not notice changes until significant damage
- This is why screening and regular exams are crucial

**Your Situation:**
Based on the AI analysis showing {diagnosis} with {confidence:.1f}% confidence, even if you don't have symptoms, you should see an ophthalmologist for proper evaluation.

**Report These to Your Doctor:**
- Any vision changes (even subtle)
- When symptoms started
- What makes them better or worse
- Impact on daily activities

Remember: Absence of symptoms doesn't mean absence of disease! Many eye conditions are asymptomatic in early stages.

⚕️ *Schedule an eye examination even if you feel fine.*"""
    
    # Questions about causes/why/how it happens
    elif any(word in question_lower for word in ['cause', 'why', 'how did', 'why did', 'get this', 'develop']):
        causes = current_disease_info.get('causes', 'Multiple factors can contribute.')
        return f"""**What Causes {diagnosis}:**

{causes}

**Risk Factors to Consider:**
{"**AMD:** Age (>50), smoking, family history, race (more common in Caucasians), obesity, high blood pressure, poor diet" if diagnosis == 'AMD' else ""}
{"**Diabetic Retinopathy:** Duration of diabetes, poor blood sugar control, high blood pressure, high cholesterol, pregnancy, smoking" if diagnosis == 'Diabetic Retinopathy' else ""}
{"**Cataract:** Aging (most common), diabetes, smoking, excessive alcohol, prolonged sun exposure, eye injury, certain medications (steroids), radiation exposure" if diagnosis == 'Cataract' else ""}
{"**Glaucoma:** Age (>60), family history, race (African American, Asian, Hispanic at higher risk), high eye pressure, thin corneas, eye injury, certain medical conditions" if diagnosis == 'Glaucoma' else ""}
{"No specific risk factors apply - your eyes appear healthy!" if diagnosis == 'Normal' else ""}

**Can You Prevent It?**
- Some risk factors are controllable (lifestyle)
- Some are not (age, genetics, race)
- Early detection and management can slow or prevent progression
- Regular eye exams are your best defense

**Understanding Your Result:**
The AI detected patterns consistent with {diagnosis} (confidence: {confidence:.1f}%). An ophthalmologist can:
- Confirm the diagnosis
- Identify specific risk factors in your case
- Determine what might have contributed
- Create a personalized management plan

⚕️ *Focus on what you can control going forward - consult your doctor for a personalized assessment.*"""
    
    # Handle general conversation (placed after medical questions so medical queries take priority)
    # Act like a real doctor - natural, personable, conversational
    
    # Clarifications and corrections
    if any(phrase in question_lower for phrase in ["not mine", "not my", "someone else", "not me", "another person", "other person", "picture of", "image of", "not for me"]):
        return f"Ah, I understand! You're reviewing results for someone else. In that case, this image shows {diagnosis}. I can still explain what this condition means, treatment options, and what steps they should take. What would you like to know about {diagnosis}?"
    
    # Follow-up acknowledgments
    if any(phrase in question_lower for phrase in ["i see", "i understand", "got it", "make sense", "makes sense", "i get it", "understood", "alright", "okay then"]):
        return f"Great! Do you have any other questions about the {diagnosis} result? I'm here to help explain anything you'd like to know."
    
    # Asking for more info / tell me more
    if any(phrase in question_lower for phrase in ["tell me more", "more info", "more information", "go on", "continue", "explain more", "what else", "anything else"]):
        return f"Sure! About {diagnosis} - it {'appears to be a healthy eye with no detected issues' if diagnosis == 'Normal' else 'is a condition that requires attention'}. What specific aspect would you like me to explain? I can discuss symptoms, causes, treatment, prevention, or what steps to take next. Just ask!"
    
    # Greetings
    if any(greeting in question_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']) and len(question_lower) < 30:
        greetings = [
            f"Hello! Good to see you. I've been reviewing the results - {diagnosis} was detected. How are you feeling about this?",
            f"Hi there! I just finished analyzing the retinal images. The results show {diagnosis}. Do you have any concerns you'd like to discuss?",
            f"Hello! I'm reviewing the eye scan results right now. We detected {diagnosis}. What questions do you have for me?"
        ]
        import random
        return random.choice(greetings)
    
    # How are you / personal questions
    if any(phrase in question_lower for phrase in ['how are you', 'how r u', 'whats up', "what's up", 'how do you do']):
        responses = [
            f"I'm doing well, thanks for asking! Just been reviewing results. This scan shows {diagnosis}. Anything specific you'd like to know?",
            f"I'm good, thank you! I've been quite busy today with image analysis. Speaking of which, this one shows {diagnosis}. Any questions about it?",
            f"I'm well, appreciate you asking! Right now I'm focused on this case - the analysis detected {diagnosis}. What would you like to discuss?"
        ]
        import random
        return random.choice(responses)
    
    # What are you doing
    if any(phrase in question_lower for phrase in ['what are you doing', 'what r u doing', 'whatcha doing', 'what you doing']):
        responses = [
            f"I'm currently reviewing eye examination results. The AI analysis detected {diagnosis} with {confidence:.1f}% confidence. I'm here to discuss this with you.",
            f"Just going through retinal scan data. Based on the imaging, we've identified {diagnosis}. Would you like me to explain what this means?",
            f"I'm analyzing fundus images and reviewing the findings. The system flagged {diagnosis}. Let me know what questions you have."
        ]
        import random
        return random.choice(responses)
    
    # Personal questions - food/dinner/lunch
    if any(phrase in question_lower for phrase in ['did you eat', 'have you eaten', 'did you have', 'have dinner', 'have lunch', 'have breakfast', 'ate yet']):
        return f"I don't eat - I'm an AI! But I appreciate you asking. 😊 Good nutrition is important for eye health though! Anyway, shall we discuss the {diagnosis} result?"
    
    # Busy/tired questions
    if any(phrase in question_lower for phrase in ['are you busy', 'are you tired', 'are you free']):
        return f"I'm never too busy! Right now I'm focused on this analysis. The results show {diagnosis}, and I want to make sure all questions are answered. What would you like to know?"
    
    # Where are you
    if 'where are you' in question_lower or 'where r u' in question_lower:
        return f"I'm right here in the Eye Disease Detection System, reviewing your results! The analysis shows {diagnosis}. I'm available anytime you need to discuss your eye health."
    
    # Thank you
    if any(phrase in question_lower for phrase in ['thank you', 'thanks', 'thank u', 'thx', 'appreciate']):
        return "You're very welcome! That's what I'm here for. If you have any other questions about your results or eye health, don't hesitate to ask. Take care!"
    
    # Who are you / what are you (but not "what is" which is medical)
    if any(phrase in question_lower for phrase in ['who are you', 'tell me about yourself', 'introduce yourself']) or question_lower == 'what are you':
        return f"I'm an AI medical assistant - think of me as a virtual ophthalmology consultant. I help patients understand their eye disease screening results. I've reviewed your scan, which shows {diagnosis}, and I can explain what this means, discuss treatment options, and guide you on next steps. What would you like to know?"
    
    # Help / capabilities
    if question_lower in ['help', 'help me', 'what can you do', 'capabilities']:
        return f"I'm here to help you understand your {diagnosis} diagnosis. I can explain the condition, discuss treatment options, talk about prevention, answer questions about accuracy, and guide you on what to do next. Just ask me anything - treat this like a regular doctor's consultation!"
    
    # General yes/no/ok responses
    if question_lower in ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'no', 'nope', 'nah']:
        return f"Alright. Your results show {diagnosis} with {confidence:.1f}% confidence. What would you like to discuss about this?"
    
    # Jokes / humor
    if any(word in question_lower for word in ['joke', 'funny', 'laugh', 'humor']):
        return f"Ha! I appreciate the lighter mood. Here's one: Why did the eye go to school? To improve its pupils! 😄 But let's talk about your {diagnosis} result - humor aside, I want to make sure you understand your diagnosis."
    
    # Goodbye / leaving
    if any(phrase in question_lower for phrase in ['bye', 'goodbye', 'see you', 'gotta go', 'have to go', 'talk later']):
        return f"Take care! Remember to {'keep up the good work with your eye health' if diagnosis == 'Normal' else 'schedule that appointment with an ophthalmologist'}. Feel free to come back anytime you have questions. Stay healthy!"
    
    # Confused / don't understand
    if any(phrase in question_lower for phrase in ["don't understand", "confused", "what do you mean", "explain better", "clarify"]):
        return f"Let me put it more simply: Your eye scan shows {diagnosis}. {'That means your eyes look healthy!' if diagnosis == 'Normal' else 'This is a condition that needs attention.'} What part would you like me to explain in simpler terms?"
    
    # Weather/time/date
    if any(word in question_lower for word in ['weather', 'temperature', 'hot', 'cold', 'rain', 'sunny']):
        return f"I can't check the weather from here, but I hope it's nice where you are! Now, about your eye health - your results show {diagnosis}. Should we discuss that?"
    
    # Generic "what" questions that aren't medical
    if question_lower in ['what', 'what?', 'huh', 'huh?', 'what happened']:
        return f"Your eye examination results came back showing {diagnosis} with {confidence:.1f}% confidence. I'm here to explain what this means and answer your questions. What would you like to know?"
    
    # If nothing matched, return helpful default
    return f"""I'm here to help with your eye health! Your analysis shows **{diagnosis}** with {confidence:.1f}% confidence.

I can answer questions like:
- "What is {diagnosis}?" or "Tell me about this condition"
- "How is this treated?" or "What are my treatment options?"
- "What should I do next?" or "What's my action plan?"
- "What causes this?" or "How did I get this?"
- "How can I prevent this?" or "What lifestyle changes help?"

Or just chat naturally - ask me anything in your own words! I'm here to help you understand your results."""

def get_interpretation(predicted_class, confidence, model_type="amd"):
    """Get medical interpretation"""
    if model_type == "glaucoma":
        if predicted_class == "Normal":
            return "✓ No glaucoma detected. The fundus appears healthy.", "normal"
        else:  # Glaucoma
            return "⚠️ Glaucoma detected. Recommend immediate specialist consultation for optic nerve assessment.", "abnormal"
    else:  # AMD model
        if predicted_class == "Normal":
            return "✓ No abnormalities detected. The fundus appears healthy.", "normal"
        elif predicted_class == "AMD":
            return "⚠️ Age-Related Macular Degeneration (AMD) detected. Recommend specialist consultation.", "abnormal"
        elif predicted_class == "Cataract":
            return "⚠️ Cataract detected. Recommend ophthalmologist consultation.", "abnormal"
        else:  # Diabetic Retinopathy
            return "⚠️ Diabetic Retinopathy detected. Recommend immediate specialist consultation.", "abnormal"

# Load models at startup
amd_model, amd_accuracy = load_amd_model()
glaucoma_model, glaucoma_accuracy = load_glaucoma_model()

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Eye Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Fundus Image Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This system can detect:
        - **AMD** (Age-Related Macular Degeneration)
        - **Diabetic Retinopathy**
        - **Cataract**
        - **Glaucoma**
        - **Normal/Healthy** eyes
        """)
        
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. Upload a fundus image
        2. Click "Analyze Image"
        3. View prediction results
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This is an AI-assisted tool for educational purposes. 
        Always consult a medical professional for diagnosis.
        """)
        
        # Chat history management
        if st.session_state.get('results_available', False) and st.session_state.get('chat_history'):
            st.markdown("---")
            st.header("💬 Chat")
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            st.caption(f"{len(st.session_state.get('chat_history', []))} messages")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a fundus image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a retinal fundus photograph"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with all models..."):
                    # Run both models independently
                    amd_pred, amd_conf, amd_results = predict_amd(image)
                    
                    # Run Glaucoma model if available
                    glaucoma_pred, glaucoma_conf, glaucoma_results = None, None, None
                    if glaucoma_model is not None:
                        glaucoma_pred, glaucoma_conf, glaucoma_results = predict_glaucoma(image)
                    
                    # Store results in session state
                    st.session_state.amd_prediction = amd_pred
                    st.session_state.amd_confidence = amd_conf
                    st.session_state.amd_results = amd_results
                    st.session_state.glaucoma_prediction = glaucoma_pred
                    st.session_state.glaucoma_confidence = glaucoma_conf
                    st.session_state.glaucoma_results = glaucoma_results
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'amd_prediction' in st.session_state:
            amd_pred = st.session_state.amd_prediction
            amd_conf = st.session_state.amd_confidence
            amd_results = st.session_state.amd_results
            glaucoma_pred = st.session_state.glaucoma_prediction
            glaucoma_conf = st.session_state.glaucoma_confidence
            glaucoma_results = st.session_state.glaucoma_results
            
            # Determine overall status and findings
            detected_diseases = []
            all_confidences = []
            
            # Check AMD model results
            if amd_pred is not None and amd_pred != "Normal":
                detected_diseases.append(amd_pred)
                all_confidences.append(amd_conf)
            
            # Check Glaucoma model results ONLY if AMD model shows Normal
            # (because glaucoma model can't distinguish between AMD/DR/Cataract and just sees them as abnormal)
            if glaucoma_pred is not None and glaucoma_pred == "Glaucoma":
                # Only trust glaucoma results if AMD model thinks it's normal
                if amd_pred == "Normal" or amd_results.get("Normal", 0) > 0.5:
                    detected_diseases.append("Glaucoma")
                    all_confidences.append(glaucoma_conf)
            
            # Main Prediction Box
            if len(detected_diseases) == 0:
                # All normal
                st.markdown('<div class="prediction-box normal">', unsafe_allow_html=True)
                st.subheader("✓ Healthy Eye Detected")
                st.markdown("**Status:** Normal")
                st.markdown("No abnormalities detected. The fundus appears healthy.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store results for chatbot
                st.session_state.results_available = True
                st.session_state.diagnosis = "Normal"
                st.session_state.confidence = 100.0
                st.session_state.all_results = {**amd_results, **glaucoma_results} if glaucoma_results else amd_results
            else:
                # Find disease with highest confidence
                max_idx = all_confidences.index(max(all_confidences))
                primary_disease = detected_diseases[max_idx]
                primary_confidence = all_confidences[max_idx]
                
                # Store results for chatbot
                st.session_state.results_available = True
                st.session_state.diagnosis = primary_disease
                st.session_state.confidence = primary_confidence * 100
                st.session_state.all_results = {**amd_results, **glaucoma_results} if glaucoma_results else amd_results
                
                # Disease detected
                st.markdown('<div class="prediction-box abnormal">', unsafe_allow_html=True)
                st.subheader("⚠️ Abnormality Detected")
                st.markdown(f"**Detected Disease:** {primary_disease}")
                st.markdown(f"**Confidence:** {primary_confidence*100:.2f}%")
                st.markdown("\n**⚠️ Recommendation:** Consult with an ophthalmologist for comprehensive evaluation and treatment.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Scroll to results section
            st.markdown("""
                <script>
                    window.parent.document.querySelector('section[data-testid="stAppViewContainer"]').scrollTop = 0;
                </script>
            """, unsafe_allow_html=True)
            
            # Confidence Level
            if len(all_confidences) > 0:
                avg_confidence = sum(all_confidences) / len(all_confidences)
                st.subheader("Confidence Level")
                if avg_confidence > 0.9:
                    st.success("High Confidence (>90%)")
                elif avg_confidence > 0.7:
                    st.warning("Moderate Confidence (70-90%)")
                else:
                    st.error("Low Confidence (<70%) - Manual review recommended")
            
            # Detailed Analysis
            st.subheader("📈 Detailed Analysis")
            
            # Combine all probabilities
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**General Conditions:**")
                if amd_pred is not None:
                    for cls in AMD_CLASSES:
                        prob = amd_results[cls]*100
                        if prob > 10:  # Only show significant probabilities
                            st.write(f"• {cls}: {prob:.1f}%")
            
            with col_b:
                st.markdown("**Optic Nerve:**")
                if glaucoma_pred is not None and glaucoma_results:
                    # Only show glaucoma results if AMD model indicates normal retina
                    if amd_pred == "Normal" or amd_results.get("Normal", 0) > 0.5:
                        for cls in GLAUCOMA_CLASSES:
                            prob = glaucoma_results[cls]*100
                            st.write(f"• {cls}: {prob:.1f}%")
                    else:
                        st.info("⚠️ Glaucoma assessment unreliable when other conditions present")
                else:
                    st.info("Glaucoma model not available")
            
            # Expandable full probabilities
            with st.expander("🔍 View All Probabilities"):
                st.markdown("**AMD/Cataract/Diabetic Retinopathy Model:**")
                for cls in AMD_CLASSES:
                    st.write(f"  {cls}: {amd_results[cls]*100:.2f}%")
                
                st.markdown("\n**Glaucoma Model:**")
                if glaucoma_results:
                    # Check if results are reliable
                    if amd_pred == "Normal" or amd_results.get("Normal", 0) > 0.5:
                        for cls in GLAUCOMA_CLASSES:
                            st.write(f"  {cls}: {glaucoma_results[cls]*100:.2f}%")
                    else:
                        st.write("  ⚠️ Results unreliable (other retinal condition detected)")
                        st.write(f"  Normal: {glaucoma_results['Normal']*100:.2f}% (not reliable)")
                        st.write(f"  Glaucoma: {glaucoma_results['Glaucoma']*100:.2f}% (not reliable)")
                else:
                    st.write("  Model not available")
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Chatbot Section - only show if results exist
    if st.session_state.get('results_available', False):
        st.markdown("---")
        st.markdown("### 💬 Ask Questions About Your Results")
        st.info("Ask me anything about the diagnosis, treatment options, prevention, or what the results mean!")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_question = st.chat_input("Ask a question about your results...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Get bot response
            with st.spinner("Thinking..."):
                bot_response = get_chat_response(
                    user_question,
                    st.session_state.get('diagnosis', ''),
                    st.session_state.get('confidence', 0),
                    st.session_state.get('all_results', {})
                )
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            
            # Rerun to display new messages
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Developed using:</strong> PyTorch • ResNet50 • Transfer Learning • Streamlit</p>
    <p>For educational and research purposes only.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
