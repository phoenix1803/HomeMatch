import os
import json
import streamlit as st
import chromadb
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from typing import List, Dict

# --- Environment setup ---
load_dotenv()

# --- Configuration and initialization ---
def initialize_app():
    # Set page configuration
    st.set_page_config(
        page_title="HomeMatch - Intelligent Property Finder",
        page_icon="🏠",
        layout="wide",
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .property-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .price-tag {
        font-size: 24px;
        font-weight: bold;
        color: #2c7873;
    }
    .neighborhood-name {
        font-size: 20px;
        color: #333;
        margin-bottom: 10px;
    }
    .stMultiSelect [data-baseweb=select] span {
        max-width: 300px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LLM and Embedding Setup ---
@st.cache_resource
def load_llm_components():
    """Load and cache LLM components to avoid reinitialization"""
    llm = OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://openai.vocareum.com/v1"
    )

    text_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://openai.vocareum.com/v1"
    )
    
    return llm, text_embeddings
import random
import re
from datetime import datetime
# --- Database Setup ---
@st.cache_resource
def setup_database():
    """Initialize ChromaDB with persistent storage"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="properties_semantic",
        metadata={"hnsw:space": "cosine"}
    )
    return collection
import uuid
import time
# --- Data Generation ---
def generate_property_listings(n: int = 150) -> List[Dict]:
    """Generate diverse synthetic property listings using LLM or fallback"""
    llm, _ = load_llm_components()

    neighborhoods = [
        "Downtown Core", "Riverside Heights", "Oakwood Gardens", 
        "Highland Park", "Westside Village", "Harbor District",
        "University Heights", "Maple Ridge", "Sunset Hills"
    ]

    prompt = f"""Generate {n} diverse real estate listings as a JSON array. Each entry should contain:
    - id (string, unique identifier)
    - neighborhood (string, use varied neighborhoods including: {', '.join(neighborhoods)})
    - price (integer between 150000 and 2000000)
    - bedrooms (integer between 1 and 6)
    - bathrooms (integer or float between 1 and 4.5)
    - size_sqft (integer between 500 and 4500)
    - year_built (integer between 1950 and 2023)
    - property_type (string, one of: "Single Family Home", "Condo", "Townhouse", "Apartment")
    - description (string, 3-4 detailed sentences about property features)
    - neighborhood_description (string, 1-2 sentences about location benefits)
    - amenities (array of strings, 4-8 features like "Hardwood Floors", "Granite Countertops", etc.)
    
    Make listings varied in style, price range, size, and features.
    Format as valid JSON that can be parsed directly.
    """

    try:
        response = llm.invoke(prompt)
        json_content = response.strip()
        if not json_content.startswith('['):
            json_match = re.search(r'\[\s*{.*}\s*\]', json_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
        return json.loads(json_content)

    except Exception as e:
        st.error(f"Error generating listings: {str(e)}")
        
        # Simulate latency
        time.sleep(1.2)

        # Random fallback generator
        property_types = ["Single Family Home", "Condo", "Townhouse", "Apartment"]
        amenities_pool = [
            "Hardwood Floors", "Granite Countertops", "In-unit Laundry",
            "Fitness Center", "Updated Kitchen", "Deck", "Fenced Yard",
            "Attached Garage", "Central AC", "Walk-in Closets",
            "Fireplace", "Swimming Pool", "Smart Home Features"
        ]
        descriptions_start = [
            "Spacious and inviting", "Modern and well-maintained", 
            "Charming and recently renovated", "Bright and airy", 
            "Elegant and thoughtfully designed", "Stylish and functional"
        ]
        features_snippets = [
            "with an open floor plan and abundant natural light",
            "featuring a chef's kitchen with stainless steel appliances",
            "boasts hardwood floors and a cozy fireplace",
            "with updated bathrooms and ample storage",
            "including a finished basement and private backyard",
            "offering panoramic views and walk-in closets"
        ]
        neighborhood_snippets = [
            "Close to schools, parks, and shopping centers.",
            "Perfect for commuters with easy transit access.",
            "Located in a vibrant area with great nightlife.",
            "Quiet streets and family-friendly atmosphere.",
            "Access to scenic walking trails and green spaces.",
            "Bustling neighborhood with cafes and boutiques."
        ]

        listings = []
        for _ in range(150):
            listing = {
                "id": str(uuid.uuid4()),
                "neighborhood": random.choice(neighborhoods),
                "price": random.randint(150_000, 2_000_000),
                "bedrooms": random.randint(1, 6),
                "bathrooms": round(random.uniform(1, 4.5), 1),
                "size_sqft": random.randint(500, 4500),
                "year_built": random.randint(1950, 2023),
                "property_type": random.choice(property_types),
                "description": f"{random.choice(descriptions_start)} {random.choice(features_snippets)}. "
                               f"{random.choice(features_snippets)}. {random.choice(features_snippets)}.",
                "neighborhood_description": random.choice(neighborhood_snippets),
                "amenities": random.sample(amenities_pool, random.randint(4, 8))
            }
            listings.append(listing)

        return listings
def generate_properties_listings(n: int = 150) -> List[Dict]:
    """Generate random property listings using predefined options"""
    neighborhoods = [
        "Downtown Core", "Riverside Heights", "Oakwood Gardens",
        "Highland Park", "Westside Village", "Harbor District",
        "University Heights", "Maple Ridge", "Sunset Hills"
    ]
    
    property_types = [
        "Single Family Home", "Condo", "Townhouse", "Apartment"
    ]
    
    amenities_pool = [
        "Hardwood Floors", "Granite Countertops", "Stainless Steel Appliances",
        "In-unit Laundry", "Fitness Center", "Swimming Pool", "Balcony",
        "Central AC", "Smart Home System", "Walk-in Closet", "Fireplace",
        "Gourmet Kitchen", "Solar Panels", "Roof Deck", "Wine Cellar"
    ]
    
    listings = []
    
    for i in range(150):
        prop_id = f"prop_{i+1:03d}"
        neighborhood = random.choice(neighborhoods)
        property_type = random.choice(property_types)
        
        # Generate random features
        price = random.randint(150000, 2000000)
        bedrooms = random.randint(1, 6)
        bathrooms = round(random.uniform(1.0, 4.5), 1)
        size_sqft = random.randint(500, 4500)
        year_built = random.randint(1950, datetime.now().year)
        
        # Generate descriptions
        description = f"A {random.choice(['beautiful', 'spacious', 'modern'])} {property_type.lower()} " \
                      f"with {random.randint(2,5)} bedrooms and {bathrooms} bathrooms. " \
                      f"Features include {random.choice(['an open floor plan', 'vaulted ceilings', 'large windows'])} " \
                      f"and {random.choice(['a recently updated kitchen', 'new flooring', 'energy-efficient appliances'])}."
        
        neighborhood_desc = f"Located in the {neighborhood} area, close to " \
                            f"{random.choice(['excellent schools', 'shopping centers', 'public transportation', 'parks'])} " \
                            f"and {random.choice(['trendy restaurants', 'cultural attractions', 'hiking trails', 'entertainment venues'])}."
        
        # Select random amenities
        amenities = random.sample(amenities_pool, k=random.randint(4, 8))
        
        listing = {
            "id": prop_id,
            "neighborhood": neighborhood,
            "price": price,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "size_sqft": size_sqft,
            "year_built": year_built,
            "property_type": property_type,
            "description": description,
            "neighborhood_description": neighborhood_desc,
            "amenities": amenities
        }
        
        listings.append(listing)
    
    return listings
LISTINGS_FILE = "Listings.txt"

def save_listings_to_file(listings: List[Dict], path: str = LISTINGS_FILE):
    """
    Write the raw JSON array of listings to a .txt file.
    If the LLM fails and you hit the fallback, this still writes that fallback.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            # Pretty-print so it's human-readable
            json.dump(listings, f, indent=2, ensure_ascii=False)
        st.info(f"✅ Wrote {len(listings)} listings to {path}")
    except Exception as e:
        st.error(f"Failed to write listings file: {e}")

# --- Database Management ---
def initialize_database():
    """Initialize database with random listings"""
    collection = setup_database()
    
    # Check if database is already populated
    if len(collection.get()['ids']) == 0:
        # Generate random listings
        listings = generate_property_listings(150)
        
        # Add listings to database
        for prop in listings:
            prop_id = prop.get('id', str(hash(prop['description']))[0:8])
            
            # Create rich text representation for semantic search
            rich_text = f"""
            {prop['property_type']} in {prop['neighborhood']}. 
            {prop['description']} 
            {prop['neighborhood_description']}
            Features: {', '.join(prop['amenities'])}
            """

            # Get embedding
            _, embeddings = load_llm_components()
            embedding = embeddings.embed_query(rich_text)

            # Prepare metadata: serialize amenities list into a comma-separated string
            meta = prop.copy()
            meta['amenities'] = ', '.join(prop['amenities'])

            # Add to collection
            collection.add(
                ids=[prop_id],
                embeddings=[embedding],
                documents=[rich_text],
                metadatas=[meta]
            )
    
    return


# --- Search Functionality ---
def search_properties(query_text: str, filters: Dict = None):
    """Search properties with semantic search and optional filters"""
    collection = setup_database()
    _, embeddings = load_llm_components()
    
    # Create query embedding
    query_embedding = embeddings.embed_query(query_text)
    
    # Execute search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["metadatas", "documents", "distances"]
    )
    
    # Convert to DataFrame for easier filtering
    df_results = pd.DataFrame({
        'id': results['ids'][0],
        'metadata': results['metadatas'][0],
        'document': results['documents'][0],
        'relevance': [1 - d for d in results['distances'][0]]  # Convert distance to relevance score
    })
    
    # Apply filters if provided
    if filters:
        if filters.get('min_price'):
            df_results = df_results[df_results['metadata'].apply(lambda x: x['price'] >= filters['min_price'])]
        if filters.get('max_price'):
            df_results = df_results[df_results['metadata'].apply(lambda x: x['price'] <= filters['max_price'])]
        if filters.get('min_beds'):
            df_results = df_results[df_results['metadata'].apply(lambda x: x['bedrooms'] >= filters['min_beds'])]
        if filters.get('property_types') and len(filters['property_types']) > 0:
            df_results = df_results[df_results['metadata'].apply(
                lambda x: x['property_type'] in filters['property_types']
            )]
        if filters.get('neighborhoods') and len(filters['neighborhoods']) > 0:
            df_results = df_results[df_results['metadata'].apply(
                lambda x: x['neighborhood'] in filters['neighborhoods']
            )]
    
    return df_results.sort_values('relevance', ascending=False)

# --- Personalization ---
def generate_personalized_description(property_data: Dict, user_query: str) -> str:
    """Create personalized property highlights based on search criteria"""
    llm, _ = load_llm_components()
    
    prompt = f"""As a real estate expert, highlight features of this property that would appeal to someone looking for: '{user_query}'.
    Focus on matching relevant aspects while being honest and accurate.
    
    Property details:
    - {property_data['property_type']} in {property_data['neighborhood']}
    - ${property_data['price']:,}, {property_data['bedrooms']} bed, {property_data['bathrooms']} bath
    - {property_data['size_sqft']} sqft, built in {property_data['year_built']}
    - Description: {property_data['description']}
    - Neighborhood: {property_data['neighborhood_description']}
    - Amenities: {', '.join(property_data['amenities'])}
    
    Write a concise, personalized highlight paragraph (3-4 sentences) showing why this matches their needs. Don't just repeat property details.
    """
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        st.warning(f"Couldn't generate personalized description: {str(e)}")
        return property_data['description']

# --- Streamlit UI ---
def display_search_interface():
    """Display the main search interface"""
    st.title("🏠 HomeMatch - Intelligent Property Finder")
    
    # Initialize database
    initialize_database()
    collection = setup_database()
    
    # Get all properties for filter options
    all_properties = collection.get(include=["metadatas"])
    
    if len(all_properties['ids']) == 0:
        st.warning("No properties in database. Please add some properties first.")
        return
    if st.button("Export Current Listings"):
            collection = setup_database()
            all_properties = collection.get(include=["metadatas"])
            
            if len(all_properties['ids']) == 0:
                st.warning("No properties to export!")
            else:
                # Convert metadata back to original format
                listings = []
                for meta in all_properties['metadatas']:
                    # Create copy and fix amenities format
                    corrected = meta.copy()
                    corrected['amenities'] = meta['amenities'].split(', ')
                    listings.append(corrected)
                
                # Create JSON data
                json_data = json.dumps(listings, indent=2)
                
                # Offer download
                st.download_button(
                    label="Download Listings as JSON",
                    data=json_data,
                    file_name="property_listings.json",
                    mime="application/json",
                    help="Download all current property listings in JSON format"
                )
    # Extract unique values for filters
    all_metadata = all_properties['metadatas']
    neighborhoods = sorted(list(set(p['neighborhood'] for p in all_metadata)))
    property_types = sorted(list(set(p['property_type'] for p in all_metadata)))
    
    # Layout with sidebar
    with st.sidebar:
        st.header("Search Criteria")
        
        search_text = st.text_area(
            "Describe your ideal home:",
            "Modern home with open floor plan and updated kitchen near parks",
            height=100,
            help="Describe features, location preferences, or lifestyle needs"
        )
        
        st.subheader("Filters")
        
        # Price range
        cols = st.columns(2)
        with cols[0]:
            min_price = st.number_input("Min Price", 
                                        min_value=100000, 
                                        max_value=2000000, 
                                        value=200000,
                                        step=50000)
        with cols[1]:
            max_price = st.number_input("Max Price", 
                                        min_value=100000, 
                                        max_value=5000000, 
                                        value=1000000,
                                        step=50000)
        
        # Bedrooms
        min_beds = st.slider("Minimum Bedrooms", 1, 6, 2)
        
        # Property type
        property_type_filter = st.multiselect("Property Type", 
                                             property_types,
                                             default=[])
        
        # Neighborhoods
        neighborhood_filter = st.multiselect("Neighborhoods", 
                                            neighborhoods,
                                            default=[])
        
        filters = {
            'min_price': min_price,
            'max_price': max_price,
            'min_beds': min_beds,
            'property_types': property_type_filter,
            'neighborhoods': neighborhood_filter
        }
        
        # Search button
        search_button = st.button("🔍 Find Matching Properties", type="primary", use_container_width=True)
    
    # Main content area
    if search_button or ('search_results' in st.session_state):
        with st.spinner("Searching for your perfect home..."):
            # Perform search
            results = search_properties(search_text, filters)
            
            # Store in session state
            st.session_state.search_results = results
            st.session_state.search_query = search_text
    
    # Display results if available
    if 'search_results' in st.session_state:
        results = st.session_state.search_results
        
        if len(results) == 0:
            st.info("No properties match your criteria. Try broadening your search or changing filters.")
        else:
            st.success(f"Found {len(results)} matching properties")
            
            # Results summary
            summary_cols = st.columns([1, 1, 1])
            with summary_cols[0]:
                avg_price = int(results['metadata'].apply(lambda x: x['price']).mean())
                st.metric("Average Price", f"${avg_price:,}")
            with summary_cols[1]:
                avg_size = int(results['metadata'].apply(lambda x: x['size_sqft']).mean())
                st.metric("Average Size", f"{avg_size} sqft")
            with summary_cols[2]:
                neighborhoods = results['metadata'].apply(lambda x: x['neighborhood']).nunique()
                st.metric("Neighborhoods", neighborhoods)
            
            st.divider()
            
            # Display individual results
            for idx, (_, row) in enumerate(results.head(5).iterrows()):
                property_data = row['metadata']
                
                # Property card
                with st.container():
                    st.markdown(f"""<div class="property-card">
                        <div class="neighborhood-name">
                            {property_data['property_type']} in {property_data['neighborhood']}
                        </div>
                    </div>""", unsafe_allow_html=True)
                    
                    cols = st.columns([2, 3])
                    
                    with cols[0]:
                        # Property details
                        st.markdown(f"""
                        <div class="price-tag">${property_data['price']:,}</div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        **{property_data['bedrooms']}** bed | **{property_data['bathrooms']}** bath | **{property_data['size_sqft']}** sqft  
                        Built: {property_data['year_built']}
                        """)
                        
                        # Amenities as tags
                        amenities_html = " ".join([f"<span style='background-color: #e1f5fe; padding: 3px 8px; border-radius: 12px; margin-right: 5px; font-size: 0.8em;'>{a}</span>" for a in property_data['amenities'][:5]])
                        st.markdown(f"<div style='margin-top: 10px;'>{amenities_html}</div>", unsafe_allow_html=True)
                    
                    with cols[1]:
                        # Generate personalized description on demand
                        with st.spinner("Generating personalized highlights..."):
                            personalized = generate_personalized_description(
                                property_data, 
                                st.session_state.search_query
                            )
                        
                        st.markdown("### Why this might be perfect for you:")
                        st.markdown(f"*{personalized}*")
                        
                        # Match score visualization
                        match_percentage = int(row['relevance'] * 100)
                        st.progress(row['relevance'], text=f"Match Score: {match_percentage}%")
                    
                    # Property details expander
                    with st.expander("View full property details"):
                        st.markdown(f"**Description:** {property_data['description']}")
                        st.markdown(f"**Neighborhood:** {property_data['neighborhood_description']}")
                        
                        # Display all amenities
                        st.markdown("**All Amenities:**")
                        st.markdown(", ".join(property_data['amenities']))
                
                st.divider()

# --- Main App ---
def main():
    # Initialize app settings
    initialize_app()
    
    # Session state initialization
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Display main interface
    display_search_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "HomeMatch uses advanced semantic search to find properties that match your specific needs. "
        "Try describing your ideal home in natural language for best results."
    )

if __name__ == "__main__":
    main()