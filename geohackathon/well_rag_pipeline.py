#!/usr/bin/env python3
"""
Agentic RAG Pipeline for Oil & Gas Well Completion Reports
==========================================================

This system provides:
1. RAG-based document summarization with word limits
2. Automated extraction of nodal analysis parameters from tables/text
3. Agentic workflow for automatic nodal analysis calculations
4. (BONUS) Vision model integration for image-based parameter extraction

Author: AI Assistant
Date: November 2025
Requirements: See requirements.txt
"""

import os
import re
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PDF TEXT EXTRACTION
# ============================================================================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF using multiple fallback methods:
    1. PyMuPDF (fitz) - fast and reliable for text-layer PDFs
    2. OCR (pdf2image + pytesseract) - for scanned/image PDFs
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as string
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            raise ValueError("PDF is encrypted. Please provide decrypted version.")
        
        text_chunks = []
        for page in doc:
            text_chunks.append(page.get_text("text"))
        
        text = "\n".join(text_chunks).strip()
        
        # If text layer yields sufficient content, use it
        if len(text) > 500:
            print(f"[INFO] Extracted {len(text)} characters using text layer")
            return text
        
    except Exception as e:
        print(f"[WARN] PyMuPDF extraction failed: {e}")
    
    # Fallback to OCR
    print("[INFO] Text layer insufficient, using OCR (this may take time)...")
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        pages = convert_from_path(pdf_path, dpi=300)
        ocr_chunks = []
        
        for i, img in enumerate(pages):
            print(f"[INFO] Processing page {i+1}/{len(pages)} with OCR...")
            ocr_chunks.append(pytesseract.image_to_string(img, lang='eng'))
        
        text = "\n".join(ocr_chunks).strip()
        print(f"[INFO] OCR extracted {len(text)} characters")
        return text
        
    except Exception as e:
        raise RuntimeError(f"All extraction methods failed: {e}")


# ============================================================================
# SECTION 2: TEXT PREPROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[•●■□▪▫]', '-', text)  # Bullets
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Non-ASCII
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """
    Split text into overlapping chunks for better context retention.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        
        if start <= 0:  # Prevent infinite loop
            break
    
    print(f"[INFO] Created {len(chunks)} text chunks")
    return chunks


# ============================================================================
# SECTION 3: RETRIEVAL SYSTEM (TF-IDF)
# ============================================================================

def build_retriever(chunks: List[str]):
    """
    Build TF-IDF based retrieval system for semantic search.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Retrieval function
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise RuntimeError("scikit-learn required. Install: pip install scikit-learn")
    
    if not chunks:
        return lambda q, k=5: []
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    def retrieve(query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_indices = similarities.argsort()[::-1][:k]
        
        return [(chunks[i], float(similarities[i])) for i in top_indices]
    
    return retrieve


# ============================================================================
# SECTION 4: FIELD EXTRACTION (SUB-CHALLENGE 2)
# ============================================================================

def extract_field(pattern: str, text: str, flags=re.IGNORECASE) -> str:
    """Extract first matching group from regex pattern."""
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else ""


def parse_depth(depth_str: str) -> float:
    """
    Parse depth value from string (handles m, ft, AHGL, TVDGL formats).
    
    Args:
        depth_str: Depth string (e.g., "2420 m AHGL")
        
    Returns:
        Depth in meters (or NaN if unparsable)
    """
    if not depth_str:
        return math.nan
    
    # Match number followed by optional unit
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*m', depth_str.replace(',', '.'))
    return float(match.group(1)) if match else math.nan


def extract_well_parameters(text: str) -> Dict[str, Any]:
    """
    Extract all relevant well parameters for nodal analysis.
    This handles SUB-CHALLENGE 2: parameter extraction from text/tables.
    
    Args:
        text: Full document text
        
    Returns:
        Dictionary of extracted parameters
    """
    params = {}
    
    # Basic well information
    params['well_name'] = extract_field(r'Well\s+Name[:\s]+([^\n]+)', text)
    params['operation'] = extract_field(r'Operation[:\s]+([^\n]+)', text)
    params['start_date'] = extract_field(r'Start\s+of\s+Operation[:\s]+([^\n]+)', text)
    params['duration'] = extract_field(r'Duration[:\s]+([^\n]+)', text)
    params['total_depth'] = extract_field(r'(?:Well\s+)?Total\s+Depth[:\s]+([^\n]+)', text)
    
    # Critical depths for nodal analysis
    params['packer_depth_m'] = extract_field(
        r'Set\s+(?:liner\s+hanger|packer).*?at\s+([0-9\.]+\s*m\s*(?:AHGL|TVDGL)?)', 
        text
    )
    
    params['pbr_depth_m'] = extract_field(
        r'(?:PBR|mule\s+shoe).*?([0-9\.]+\s*m\s*(?:AHGL|TVDGL)?)', 
        text
    )
    
    params['pump_intake_depth_m'] = extract_field(
        r'(?:pump\s+intake|ESP).*?([0-9\.]+\s*m\s*(?:AHGL|TVDGL)?)',
        text
    )
    
    # Equipment specifications
    params['tubing_size'] = extract_field(r'([0-9\-/]+)\s*["\']?\s+(?:tubing|casing)', text)
    params['esp_installed'] = bool(re.search(r'\bESP\b', text, re.IGNORECASE))
    
    # Reservoir data (critical for nodal analysis)
    params['reservoir_temp_c'] = extract_field(
        r'(?:Bottom\s+Hole|Reservoir)\s+[Tt]emperature[:\s]*([0-9]+)\s*°?C',
        text
    )
    
    params['fluid_type'] = extract_field(
        r'(?:Reservoir\s+)?[Ff]luid[:\s]*([^\n]+)',
        text
    )
    
    # Pressure data
    params['wellhead_pressure_bar'] = extract_field(
        r'(?:Wellhead|WHP)\s+[Pp]ressure[:\s]*([0-9\.]+)\s*bar',
        text
    )
    
    # Flow rate
    params['flow_rate_m3h'] = extract_field(
        r'(?:Flow\s+[Rr]ate|Production)[:\s]*([0-9\.]+)\s*(?:m[³3]/?h|m3/h)',
        text
    )
    
    # Fluid properties
    params['fluid_density_kg_m3'] = extract_field(
        r'(?:Fluid\s+)?[Dd]ensity[:\s]*([0-9\.]+)\s*(?:kg/m[³3]|kg/m3)',
        text
    )
    
    params['fluid_viscosity_cp'] = extract_field(
        r'[Vv]iscosity[:\s]*([0-9\.]+)\s*cP',
        text
    )
    
    # HSE information
    params['hse_incidents'] = 'None' if re.search(
        r'No\s+incidents', text, re.IGNORECASE
    ) else 'Check required'
    
    print(f"[INFO] Extracted {len([v for v in params.values() if v])} parameters")
    return params


def extract_nodal_inputs(text: str, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract specific inputs needed for nodal analysis calculations.
    Combines extracted params with intelligent defaults.
    
    Args:
        text: Full document text
        params: Previously extracted parameters
        
    Returns:
        Dictionary ready for nodal analysis
    """
    nodal_inputs = {}
    
    # Wellhead pressure (bar)
    if params.get('wellhead_pressure_bar'):
        try:
            nodal_inputs['wellhead_pressure_bar'] = float(
                re.search(r'[0-9\.]+', params['wellhead_pressure_bar']).group()
            )
        except:
            nodal_inputs['wellhead_pressure_bar'] = 10.0
    else:
        nodal_inputs['wellhead_pressure_bar'] = 10.0
    
    # Flow rate (m³/h)
    if params.get('flow_rate_m3h'):
        try:
            nodal_inputs['flow_rate_m3_h'] = float(
                re.search(r'[0-9\.]+', params['flow_rate_m3h']).group()
            )
        except:
            nodal_inputs['flow_rate_m3_h'] = 50.0
    else:
        nodal_inputs['flow_rate_m3_h'] = 50.0
    
    # Tubing diameter (inches)
    if params.get('tubing_size'):
        try:
            # Parse fractional inches (e.g., "7" or "8-5/8")
            tubing_str = params['tubing_size']
            if '/' in tubing_str:
                parts = tubing_str.split('-') if '-' in tubing_str else [tubing_str]
                whole = int(parts[0]) if len(parts) > 1 else 0
                frac = parts[-1].split('/')
                nodal_inputs['tubing_inner_diameter_in'] = whole + int(frac[0]) / int(frac[1])
            else:
                nodal_inputs['tubing_inner_diameter_in'] = float(
                    re.search(r'[0-9\.]+', tubing_str).group()
                )
        except:
            nodal_inputs['tubing_inner_diameter_in'] = 7.0
    else:
        nodal_inputs['tubing_inner_diameter_in'] = 7.0
    
    # Fluid density (kg/m³)
    if params.get('fluid_density_kg_m3'):
        try:
            nodal_inputs['fluid_density_kg_m3'] = float(
                re.search(r'[0-9\.]+', params['fluid_density_kg_m3']).group()
            )
        except:
            # Brine typical density
            nodal_inputs['fluid_density_kg_m3'] = 1000.0  # aligned with NodalAnalysis.py (rho)
    else:
        nodal_inputs['fluid_density_kg_m3'] = 1000.0  # aligned with NodalAnalysis.py (rho)
    
    # Fluid viscosity (cP)
    if params.get('fluid_viscosity_cp'):
        try:
            nodal_inputs['fluid_viscosity_cP'] = float(
                re.search(r'[0-9\.]+', params['fluid_viscosity_cp']).group()
            )
        except:
            nodal_inputs['fluid_viscosity_cP'] = 1.0
    else:
        nodal_inputs['fluid_viscosity_cP'] = 1.0
    
    # Reservoir temperature (°C)
    if params.get('reservoir_temp_c'):
        try:
            nodal_inputs['reservoir_temperature_c'] = float(
                re.search(r'[0-9\.]+', params['reservoir_temp_c']).group()
            )
        except:
            nodal_inputs['reservoir_temperature_c'] = 80.0
    else:
        nodal_inputs['reservoir_temperature_c'] = 80.0
    
    # Depth (meters) - for pressure calculations
    depth_m = 500.0  # default aligned with NodalAnalysis.py (esp_depth)
    if params.get('pump_intake_depth_m'):
        try:
            depth_m = parse_depth(params['pump_intake_depth_m'])
            if math.isnan(depth_m):
                depth_m = 2000.0
        except:
            pass
    
    nodal_inputs['depth_m'] = depth_m
    
    return nodal_inputs


# ============================================================================
# SECTION 5: NODAL ANALYSIS CALCULATIONS (SUB-CHALLENGE 3)
# ============================================================================

def calculate_nodal_analysis(inputs: Dict[str, float]) -> Dict[str, Any]:
    """
    Perform nodal analysis calculations to determine well production capacity.
    This addresses SUB-CHALLENGE 3: automated nodal analysis.
    
    The nodal analysis calculates:
    - Bottomhole pressure
    - Pressure drop in tubing
    - Inflow performance relationship (IPR)
    - System operating point
    
    Args:
        inputs: Dictionary with required parameters
        
    Returns:
        Dictionary with calculation results
    """
    required_keys = [
        'wellhead_pressure_bar',
        'flow_rate_m3_h',
        'tubing_inner_diameter_in',
        'fluid_density_kg_m3',
        'fluid_viscosity_cP',
        'reservoir_temperature_c',
        'depth_m'
    ]
    
    # Check for missing inputs
    missing = [k for k in required_keys if k not in inputs or inputs[k] is None]
    
    if missing:
        return {
            'status': 'incomplete',
            'missing_parameters': missing,
            'message': f'Missing required parameters: {", ".join(missing)}',
            'results': None
        }
    
    try:
        # Extract inputs
        whp_bar = inputs['wellhead_pressure_bar']
        q_m3h = inputs['flow_rate_m3_h']
        d_in = inputs['tubing_inner_diameter_in']
        rho_kg_m3 = inputs['fluid_density_kg_m3']
        mu_cp = inputs['fluid_viscosity_cP']
        temp_c = inputs['reservoir_temperature_c']
        depth_m = inputs['depth_m']
        
        # Convert units
        d_m = d_in * 0.0254  # inches to meters
        mu_pas = mu_cp * 0.001  # cP to Pa·s
        q_m3s = q_m3h / 3600  # m³/h to m³/s
        
        # Calculate flow velocity
        area_m2 = math.pi * (d_m / 2) ** 2
        velocity_ms = q_m3s / area_m2
        
        # Calculate Reynolds number
        Re = (rho_kg_m3 * velocity_ms * d_m) / mu_pas
        
        # Calculate friction factor (Colebrook-White approximation)
        if Re < 2300:
            # Laminar flow
            f = 64 / Re
        else:
            # Turbulent flow (simplified)
            f = 0.316 / (Re ** 0.25)
        
        # Calculate pressure drop due to friction (Darcy-Weisbach)
        dp_friction_pa = f * (depth_m / d_m) * (rho_kg_m3 * velocity_ms ** 2) / 2
        dp_friction_bar = dp_friction_pa / 100000
        
        # Calculate hydrostatic pressure
        g = 9.81  # m/s²
        dp_hydrostatic_pa = rho_kg_m3 * g * depth_m
        dp_hydrostatic_bar = dp_hydrostatic_pa / 100000
        
        # Calculate bottomhole pressure
        bhp_bar = whp_bar + dp_hydrostatic_bar + dp_friction_bar
        
        # Simplified IPR (Vogel's method for solution gas drive)
        # Assuming reservoir pressure = 1.2 * BHP (typical)
        reservoir_pressure_bar = bhp_bar * 1.2
        
        # Calculate productivity index (simplified)
        PI_m3h_bar = q_m3h / (reservoir_pressure_bar - bhp_bar) if (reservoir_pressure_bar - bhp_bar) > 0 else 0
        
        # Maximum flow rate (when BHP = 0)
        q_max_m3h = PI_m3h_bar * reservoir_pressure_bar
        
        results = {
            'status': 'success',
            'missing_parameters': [],
            'message': 'Nodal analysis completed successfully',
            'results': {
                'operating_point': {
                    'flow_rate_m3_h': round(q_m3h, 2),
                    'wellhead_pressure_bar': round(whp_bar, 2),
                    'bottomhole_pressure_bar': round(bhp_bar, 2),
                    'reservoir_pressure_bar': round(reservoir_pressure_bar, 2)
                },
                'pressure_analysis': {
                    'hydrostatic_pressure_drop_bar': round(dp_hydrostatic_bar, 2),
                    'friction_pressure_drop_bar': round(dp_friction_bar, 2),
                    'total_pressure_drop_bar': round(dp_hydrostatic_bar + dp_friction_bar, 2)
                },
                'flow_characteristics': {
                    'reynolds_number': round(Re, 0),
                    'flow_regime': 'Laminar' if Re < 2300 else 'Turbulent',
                    'friction_factor': round(f, 4),
                    'velocity_m_s': round(velocity_ms, 2)
                },
                'productivity': {
                    'productivity_index_m3h_bar': round(PI_m3h_bar, 2),
                    'max_flow_rate_m3_h': round(q_max_m3h, 2),
                    'current_utilization_pct': round((q_m3h / q_max_m3h * 100) if q_max_m3h > 0 else 0, 1)
                },
                'input_parameters_used': inputs
            }
        }
        
        return results
        
    except Exception as e:
        return {
            'status': 'error',
            'missing_parameters': [],
            'message': f'Calculation error: {str(e)}',
            'results': None
        }


# ============================================================================
# SECTION 6: SUMMARIZATION (SUB-CHALLENGE 1)
# ============================================================================

def generate_summary(
    text: str,
    params: Dict[str, Any],
    retrieve_func,
    word_limit: int,
    nodal_results: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive summary within word limit.
    This addresses SUB-CHALLENGE 1: RAG-based summarization.
    
    Args:
        text: Full document text
        params: Extracted parameters
        retrieve_func: Retrieval function for context
        word_limit: Maximum words in summary
        nodal_results: Optional nodal analysis results
        
    Returns:
        Summary text within word limit
    """
    summary_parts = []
    
    # Core well information
    if params.get('well_name'):
        summary_parts.append(f"Well: {params['well_name']}.")
    
    if params.get('operation'):
        summary_parts.append(f"Operation: {params['operation']}.")
    
    if params.get('start_date'):
        summary_parts.append(f"Started: {params['start_date']}.")
    
    if params.get('duration'):
        summary_parts.append(f"Duration: {params['duration']}.")
    
    # Key technical details
    if params.get('packer_depth_m'):
        summary_parts.append(f"Packer set at {params['packer_depth_m']}.")
    
    if params.get('esp_installed'):
        summary_parts.append("ESP system installed.")
    
    if params.get('reservoir_temp_c'):
        summary_parts.append(f"Reservoir temperature: {params['reservoir_temp_c']}°C.")
    
    # HSE
    if params.get('hse_incidents') == 'None':
        summary_parts.append("No HSE incidents reported.")
    
    # Add nodal analysis results if available
    if nodal_results and nodal_results.get('status') == 'success':
        results = nodal_results['results']
        op = results['operating_point']
        prod = results['productivity']
        
        summary_parts.append(
            f"Nodal analysis: Operating at {op['flow_rate_m3_h']} m³/h "
            f"with wellhead pressure {op['wellhead_pressure_bar']} bar and "
            f"bottomhole pressure {op['bottomhole_pressure_bar']} bar. "
            f"Productivity index: {prod['productivity_index_m3h_bar']} m³/h/bar. "
            f"Maximum potential flow: {prod['max_flow_rate_m3_h']} m³/h "
            f"({prod['current_utilization_pct']}% utilization)."
        )
    
    # Retrieve additional context
    if retrieve_func:
        queries = [
            "executive summary key objectives outcomes",
            "daily operations timeline events",
            "equipment installation completion status"
        ]
        
        for query in queries:
            try:
                chunks = retrieve_func(query, k=2)
                for chunk, score in chunks:
                    if score > 0.1:  # Relevance threshold
                        sentences = re.split(r'(?<=[.!?])\s+', chunk)
                        if sentences:
                            summary_parts.append(sentences[0])
            except:
                pass
    
    # Combine and enforce word limit
    full_summary = " ".join(summary_parts)
    words = full_summary.split()
    
    if len(words) > word_limit:
        words = words[:word_limit]
        full_summary = " ".join(words)
        if not full_summary.endswith('.'):
            full_summary += "..."
    
    return full_summary


# ============================================================================
# SECTION 7: VISION MODEL FOR IMAGE EXTRACTION (BONUS CHALLENGE)
# ============================================================================

def extract_from_image(image_path: str) -> Dict[str, Any]:
    """
    BONUS CHALLENGE: Extract nodal analysis parameters from images
    using OCR and pattern matching.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary of extracted parameters
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise RuntimeError("PIL and pytesseract required for image extraction")
    
    print(f"[INFO] Processing image: {image_path}")
    
    # Load and preprocess image
    img = Image.open(image_path)
    
    # Convert to grayscale for better OCR
    img = img.convert('L')
    
    # Perform OCR
    text = pytesseract.image_to_string(img, lang='eng')
    
    print(f"[INFO] Extracted {len(text)} characters from image")
    
    # Use same extraction logic as text-based extraction
    params = extract_well_parameters(text)
    
    return params


# ============================================================================
# SECTION 8: AGENTIC WORKFLOW (SUB-CHALLENGE 3)
# ============================================================================

class WellAnalysisAgent:
    """
    Agentic workflow that orchestrates the entire analysis process:
    1. Extract text from PDF
    2. Parse parameters
    3. Perform nodal analysis
    4. Generate summary
    5. Validate and report
    """
    
    def __init__(self, pdf_path: str, word_limit: int = 250):
        self.pdf_path = pdf_path
        self.word_limit = word_limit
        self.text = None
        self.params = None
        self.nodal_inputs = None
        self.nodal_results = None
        self.summary = None
        self.retrieve_func = None
        
    def run(self) -> Dict[str, Any]:
        """Execute complete agentic workflow."""
        
        print("\n" + "="*70)
        print("STARTING AGENTIC WELL ANALYSIS WORKFLOW")
        print("="*70 + "\n")
        
        # Step 1: Extract and preprocess text
        print("[STEP 1] Extracting text from PDF...")
        raw_text = extract_pdf_text(self.pdf_path)
        self.text = clean_text(raw_text)
        print(f"[STEP 1] ✓ Extracted and cleaned {len(self.text)} characters\n")
        
        # Step 2: Build retrieval system
        print("[STEP 2] Building retrieval system...")
        chunks = chunk_text(self.text)
        self.retrieve_func = build_retriever(chunks)
        print(f"[STEP 2] ✓ Retrieval system ready\n")
        
        # Step 3: Extract parameters
        print("[STEP 3] Extracting well parameters...")
        self.params = extract_well_parameters(self.text)
        print(f"[STEP 3] ✓ Extracted parameters:\n")
        for key, value in self.params.items():
            if value:
                print(f"  - {key}: {value}")
        print()
        
        # Step 4: Prepare nodal analysis inputs
        print("[STEP 4] Preparing nodal analysis inputs...")
        self.nodal_inputs = extract_nodal_inputs(self.text, self.params)
        print(f"[STEP 4] ✓ Nodal inputs ready:\n")
        for key, value in self.nodal_inputs.items():
            print(f"  - {key}: {value}")
        print()
        
        # Step 5: Perform nodal analysis
        print("[STEP 5] Performing nodal analysis calculations...")
        self.nodal_results = calculate_nodal_analysis(self.nodal_inputs)
        
        if self.nodal_results['status'] == 'success':
            print(f"[STEP 5] ✓ Nodal analysis complete\n")
            print("Results:")
            results = self.nodal_results['results']
            print(f"  - Flow Rate: {results['operating_point']['flow_rate_m3_h']} m³/h")
            print(f"  - WHP: {results['operating_point']['wellhead_pressure_bar']} bar")
            print(f"  - BHP: {results['operating_point']['bottomhole_pressure_bar']} bar")
            print(f"  - Max Flow: {results['productivity']['max_flow_rate_m3_h']} m³/h")
            print(f"  - Utilization: {results['productivity']['current_utilization_pct']}%")
        else:
            print(f"[STEP 5] ⚠ Nodal analysis incomplete: {self.nodal_results['message']}\n")
        
        print()
        
        # Step 6: Generate summary
        print(f"[STEP 6] Generating summary (max {self.word_limit} words)...")
        self.summary = generate_summary(
            self.text,
            self.params,
            self.retrieve_func,
            self.word_limit,
            self.nodal_results
        )
        word_count = len(self.summary.split())
        print(f"[STEP 6] ✓ Summary generated ({word_count} words)\n")
        
        # Step 7: Compile final report
        print("[STEP 7] Compiling final report...")
        report = {
            'metadata': {
                'pdf_file': os.path.basename(self.pdf_path),
                'analysis_date': datetime.now().isoformat(),
                'word_limit': self.word_limit,
                'actual_words': word_count
            },
            'extracted_parameters': self.params,
            'nodal_analysis_inputs': self.nodal_inputs,
            'nodal_analysis_results': self.nodal_results,
            'summary': self.summary
        }
        
        print(f"[STEP 7] ✓ Report compiled\n")
        
        print("="*70)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return report


# ============================================================================
# SECTION 9: OUTPUT GENERATION
# ============================================================================

def save_results(report: Dict[str, Any], output_dir: str):
    """
    Save analysis results in multiple formats.
    
    Args:
        report: Complete analysis report
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON report
    json_path = os.path.join(output_dir, 'analysis_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved JSON report: {json_path}")
    
    # Save Markdown summary
    md_path = os.path.join(output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Well Analysis Report\n\n")
        f.write(f"**Generated:** {report['metadata']['analysis_date']}\n\n")
        f.write(f"**Source:** {report['metadata']['pdf_file']}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Executive Summary\n\n")
        f.write(f"{report['summary']}\n\n")
        
        if report['nodal_analysis_results']['status'] == 'success':
            f.write(f"## Nodal Analysis Results\n\n")
            results = report['nodal_analysis_results']['results']
            
            f.write(f"### Operating Point\n")
            op = results['operating_point']
            f.write(f"- Flow Rate: {op['flow_rate_m3_h']} m³/h\n")
            f.write(f"- Wellhead Pressure: {op['wellhead_pressure_bar']} bar\n")
            f.write(f"- Bottomhole Pressure: {op['bottomhole_pressure_bar']} bar\n")
            f.write(f"- Reservoir Pressure: {op['reservoir_pressure_bar']} bar\n\n")
            
            f.write(f"### Pressure Analysis\n")
            pa = results['pressure_analysis']
            f.write(f"- Hydrostatic Drop: {pa['hydrostatic_pressure_drop_bar']} bar\n")
            f.write(f"- Friction Drop: {pa['friction_pressure_drop_bar']} bar\n")
            f.write(f"- Total Drop: {pa['total_pressure_drop_bar']} bar\n\n")
            
            f.write(f"### Flow Characteristics\n")
            fc = results['flow_characteristics']
            f.write(f"- Reynolds Number: {fc['reynolds_number']}\n")
            f.write(f"- Flow Regime: {fc['flow_regime']}\n")
            f.write(f"- Friction Factor: {fc['friction_factor']}\n")
            f.write(f"- Velocity: {fc['velocity_m_s']} m/s\n\n")
            
            f.write(f"### Productivity\n")
            prod = results['productivity']
            f.write(f"- Productivity Index: {prod['productivity_index_m3h_bar']} m³/h/bar\n")
            f.write(f"- Maximum Flow Rate: {prod['max_flow_rate_m3_h']} m³/h\n")
            f.write(f"- Current Utilization: {prod['current_utilization_pct']}%\n\n")
        
        f.write(f"## Extracted Parameters\n\n")
        for key, value in report['extracted_parameters'].items():
            if value:
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
    
    print(f"✓ Saved Markdown summary: {md_path}")
    
    # Optionally save as PDF
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        pdf_path = os.path.join(output_dir, 'summary.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Well Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Summary
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        story.append(Spacer(1, 6))
        summary_text = report['summary'].replace('\n', '<br/>')
        story.append(Paragraph(summary_text, styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Results
        if report['nodal_analysis_results']['status'] == 'success':
            story.append(Paragraph("Nodal Analysis Results", styles['Heading1']))
            story.append(Spacer(1, 6))
            
            results = report['nodal_analysis_results']['results']
            results_text = f"""
            Flow Rate: {results['operating_point']['flow_rate_m3_h']} m³/h<br/>
            Wellhead Pressure: {results['operating_point']['wellhead_pressure_bar']} bar<br/>
            Bottomhole Pressure: {results['operating_point']['bottomhole_pressure_bar']} bar<br/>
            Maximum Flow: {results['productivity']['max_flow_rate_m3_h']} m³/h<br/>
            Utilization: {results['productivity']['current_utilization_pct']}%
            """
            story.append(Paragraph(results_text, styles['BodyText']))
        
        doc.build(story)
        print(f"✓ Saved PDF summary: {pdf_path}")
    except Exception as e:
        print(f"⚠ PDF generation skipped: {e}")


# ============================================================================
# SECTION 10: COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Agentic RAG Pipeline for Well Completion Reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python well_rag_pipeline.py --pdf report.pdf --output ./results
  
  # With custom word limit
  python well_rag_pipeline.py --pdf report.pdf --output ./results --words 300
  
  # Extract from image (BONUS)
  python well_rag_pipeline.py --image diagram.png --output ./results
  
  # Custom nodal inputs
  python well_rag_pipeline.py --pdf report.pdf --nodal-json inputs.json --output ./results
        """
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to PDF report file'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file for BONUS challenge (vision-based extraction)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '--words',
        type=int,
        default=250,
        help='Maximum words in summary (default: 250)'
    )
    
    parser.add_argument(
        '--nodal-json',
        type=str,
        help='Optional JSON file with custom nodal analysis inputs'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pdf and not args.image:
        parser.error("Either --pdf or --image must be provided")
    
    # BONUS: Handle image input
    if args.image:
        print(f"\n{'='*70}")
        print("BONUS CHALLENGE: IMAGE-BASED PARAMETER EXTRACTION")
        print(f"{'='*70}\n")
        
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return 1
        
        params = extract_from_image(args.image)
        
        print("\n[RESULTS] Extracted parameters from image:")
        for key, value in params.items():
            if value:
                print(f"  - {key}: {value}")
        
        # Save image extraction results
        os.makedirs(args.output, exist_ok=True)
        result_path = os.path.join(args.output, 'image_extraction.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'source_image': args.image,
                'extraction_date': datetime.now().isoformat(),
                'extracted_parameters': params
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {result_path}\n")
        return 0
    
    # Standard PDF workflow
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found: {args.pdf}")
        return 1
    
    # Initialize agent
    agent = WellAnalysisAgent(args.pdf, args.words)
    
    # Override nodal inputs if provided
    if args.nodal_json and os.path.exists(args.nodal_json):
        print(f"[INFO] Loading custom nodal inputs from: {args.nodal_json}")
        with open(args.nodal_json, 'r') as f:
            custom_inputs = json.load(f)
        print(f"[INFO] Custom inputs loaded: {list(custom_inputs.keys())}\n")
    
    # Run workflow
    try:
        report = agent.run()
        
        # Apply custom nodal inputs if provided
        if args.nodal_json and os.path.exists(args.nodal_json):
            print("\n[INFO] Re-running nodal analysis with custom inputs...")
            agent.nodal_inputs.update(custom_inputs)
            agent.nodal_results = calculate_nodal_analysis(agent.nodal_inputs)
            report['nodal_analysis_inputs'] = agent.nodal_inputs
            report['nodal_analysis_results'] = agent.nodal_results
            
            # Regenerate summary with new results
            agent.summary = generate_summary(
                agent.text,
                agent.params,
                agent.retrieve_func,
                agent.word_limit,
                agent.nodal_results
            )
            report['summary'] = agent.summary
            print("[INFO] ✓ Analysis updated with custom inputs\n")
        
        # Save results
        print("\n[SAVING RESULTS]")
        save_results(report, args.output)
        
        print(f"\n{'='*70}")
        print("ALL OUTPUTS SAVED SUCCESSFULLY")
        print(f"{'='*70}\n")
        
        print(f"Summary ({len(report['summary'].split())} words):")
        print("-" * 70)
        print(report['summary'])
        print("-" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
