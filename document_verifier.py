"""
Enhanced Document Verification Module
Detects forgery, tampering, edited details, image manipulation, and expiry dates
"""

import os
import json
import re
from datetime import datetime, timedelta
from PIL import Image
import PyPDF2
from docx import Document
import numpy as np
from io import BytesIO
import mimetypes
import hashlib
import base64
from typing import Dict, List, Tuple, Any
import warnings

# Suppress PIL warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import advanced forensics libraries (optional)
try:
    from PIL.PngImagePlugin import PngInfo
    HAS_PNG_INFO = True
except:
    HAS_PNG_INFO = False

try:
    import exifread
    HAS_EXIF = True
except:
    HAS_EXIF = False


class DocumentForensics:
    """Advanced forensics analysis for documents and images"""
    
    @staticmethod
    def analyze_image_metadata(image_path: str) -> Dict[str, Any]:
        """Extract and analyze image metadata for signs of tampering"""
        results = {
            'metadata': {},
            'suspicious_indicators': [],
            'manipulation_score': 0
        }
        
        try:
            img = Image.open(image_path)
            
            # Get basic image info
            results['metadata']['format'] = img.format
            results['metadata']['size'] = img.size
            results['metadata']['mode'] = img.mode
            
            # Check for EXIF data
            if HAS_EXIF:
                try:
                    with open(image_path, 'rb') as f:
                        exif_data = exifread.process_file(f, details=False)
                        if exif_data:
                            results['metadata']['exif'] = {
                                str(k): str(v) for k, v in list(exif_data.items())[:10]
                            }
                        else:
                            results['suspicious_indicators'].append(
                                "⚠️ No EXIF data found - image may have been stripped or edited"
                            )
                            results['manipulation_score'] += 15
                except:
                    pass
            
            # Check image integrity
            if img.format == 'PNG':
                try:
                    img_check = Image.open(image_path)
                    img_check.verify()
                except Exception as e:
                    results['suspicious_indicators'].append(
                        f"⚠️ PNG integrity check failed - possible corruption or tampering: {str(e)}"
                    )
                    results['manipulation_score'] += 25
            
            # Analyze image properties for inconsistencies
            results['metadata']['ppi'] = img.info.get('dpi', (72, 72))
            results['metadata']['has_transparency'] = img.mode == 'RGBA'
            
            # Check for suspicious color patterns
            if img.mode in ['RGB', 'RGBA']:
                try:
                    arr = np.array(img)
                    # Check for unusual color distributions
                    unique_colors = len(np.unique(arr.reshape(-1, arr.shape[2]), axis=0))
                    
                    if unique_colors < 100 and img.mode != 'RGBA':
                        results['suspicious_indicators'].append(
                            "⚠️ Image has unusually limited color palette - may indicate editing"
                        )
                        results['manipulation_score'] += 10
                except:
                    pass
            
        except Exception as e:
            results['error'] = str(e)
        
        return results

    @staticmethod
    def analyze_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
        """Extract and analyze PDF metadata"""
        results = {
            'metadata': {},
            'suspicious_indicators': [],
            'manipulation_score': 0
        }
        
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                
                # Get document info
                if pdf.metadata:
                    results['metadata'] = {
                        k.lstrip('/'): str(v) for k, v in pdf.metadata.items()
                    }
                    
                    # Check for suspicious creation/modification patterns
                    creation_date = results['metadata'].get('CreationDate')
                    mod_date = results['metadata'].get('ModDate')
                    
                    if creation_date and mod_date:
                        if creation_date == mod_date:
                            results['suspicious_indicators'].append(
                                "⚠️ PDF creation and modification dates are identical - unusual"
                            )
                            results['manipulation_score'] += 10
                
                # Check page count changes
                results['metadata']['page_count'] = len(pdf.pages)
                
                # Check for modified content
                if pdf.is_encrypted:
                    results['suspicious_indicators'].append(
                        "⚠️ PDF is encrypted - may indicate attempts to hide content"
                    )
                    results['manipulation_score'] += 15
                    
        except Exception as e:
            results['error'] = str(e)
        
        return results


class DateAnalyzer:
    """Analyze dates in documents for inconsistencies and expiry"""
    
    # Common date patterns
    DATE_PATTERNS = [
        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})\b',  # DD MMM YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b',  # MMM DD, YYYY
        r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',  # YYYY-MM-DD
    ]
    
    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Keywords indicating expiry information
    EXPIRY_KEYWORDS = [
        'expir', 'valid until', 'valid till', 'expires', 'exp:', 'exp. ',
        'expiration', 'end date', 'date of expiry', 'valid through'
    ]
    
    @staticmethod
    def extract_dates(text: str) -> List[Dict[str, Any]]:
        """Extract all dates from text"""
        dates = []
        
        for pattern in DateAnalyzer.DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    date_str = match.group(0)
                    dates.append({
                        'text': date_str,
                        'position': match.start(),
                        'raw': match.groups()
                    })
                except:
                    pass
        
        return dates
    
    @staticmethod
    def find_expiry_dates(text: str) -> List[Dict[str, Any]]:
        """Find dates specifically mentioned as expiry dates"""
        expiry_dates = []
        
        # Look for expiry keywords followed by dates
        for keyword in DateAnalyzer.EXPIRY_KEYWORDS:
            pattern = re.compile(
                rf'{keyword}[:\s]+(\d{{1,2}}[/-](\d{{1,2}})[/-]\d{{2,4}}|\d{{1,2}}\s+\w+\s+\d{{4}})',
                re.IGNORECASE
            )
            matches = pattern.finditer(text)
            for match in matches:
                expiry_dates.append({
                    'keyword': keyword,
                    'date': match.group(1),
                    'position': match.start()
                })
        
        return expiry_dates
    
    @staticmethod
    def validate_dates(dates: List[Dict], current_date: datetime = None) -> Dict[str, Any]:
        """Validate extracted dates for consistency and expiry"""
        if current_date is None:
            current_date = datetime.now()
        
        results = {
            'valid_dates': [],
            'suspicious_indicators': [],
            'expired_documents': [],
            'future_dates': [],
            'validity_status': 'valid'
        }
        
        for date_info in dates:
            try:
                date_text = date_info.get('text', '')
                
                # Try to parse the date
                parsed_date = None
                for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d']:
                    try:
                        parsed_date = datetime.strptime(date_text, fmt)
                        break
                    except:
                        pass
                
                if parsed_date:
                    results['valid_dates'].append(parsed_date)
                    
                    # Check for expiry
                    if 'expir' in date_text.lower() or 'valid until' in date_text.lower():
                        if parsed_date < current_date:
                            results['expired_documents'].append({
                                'date': date_text,
                                'days_expired': (current_date - parsed_date).days
                            })
                            results['suspicious_indicators'].append(
                                f"🔴 CRITICAL: Document expired on {date_text} ({(current_date - parsed_date).days} days ago)"
                            )
                            results['validity_status'] = 'expired'
                        else:
                            days_until = (parsed_date - current_date).days
                            results['suspicious_indicators'].append(
                                f"✓ Document expires in {days_until} days ({date_text})"
                            )
                    
                    # Check for future dates (suspicious for issued/created dates)
                    if parsed_date > current_date + timedelta(days=1):
                        if 'issue' not in date_text.lower() and 'valid' in date_text.lower():
                            results['future_dates'].append(date_text)
                            results['suspicious_indicators'].append(
                                f"⚠️ Document has future date: {date_text} (possible forgery)"
                            )
                            results['validity_status'] = 'suspicious_future_date'
                            
            except:
                pass
        
        return results


class SpellingAnalyzer:
    """Detect spelling and institutional terminology errors"""
    
    # Common misspellings in official documents
    COMMON_MISSPELLINGS = {
        'visas': ['visa', 'visa\'s'],
        'goverment': ['government'],
        'recieved': ['received'],
        'agrement': ['agreement'],
        'autorization': ['authorization'],
        'occured': ['occurred'],
        'writted': ['written'],
        'adress': ['address'],
        'identifaction': ['identification'],
        'recomendation': ['recommendation'],
        'departement': ['department'],
        'expresion': ['expression'],
    }
    
    # Official terminology that should appear in certain documents
    OFFICIAL_TERMS = {
        'passport': ['passport', 'travel document', 'international passport'],
        'visa': ['visa', 'entry permit', 'travel authorization'],
        'certificate': ['certificate', 'certification', 'certified'],
        'application': ['application', 'applicant', 'apply'],
    }
    
    @staticmethod
    def check_spelling(text: str) -> Dict[str, Any]:
        """Check for spelling errors"""
        results = {
            'spelling_errors': [],
            'suspicious_indicators': []
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        for misspelled, corrections in SpellingAnalyzer.COMMON_MISSPELLINGS.items():
            if misspelled in words:
                results['spelling_errors'].append({
                    'word': misspelled,
                    'corrections': corrections,
                    'severity': 'high'
                })
        
        if len(results['spelling_errors']) > 5:
            results['suspicious_indicators'].append(
                f"⚠️ Found {len(results['spelling_errors'])} spelling errors - document quality may be compromised"
            )
        
        return results
    
    @staticmethod
    def check_institutional_terms(text: str, doc_type: str = 'general') -> Dict[str, Any]:
        """Check for proper institutional terminology"""
        results = {
            'institutional_errors': [],
            'suspicious_indicators': []
        }
        
        text_lower = text.lower()
        
        # Check for proper terminology based on document type
        if doc_type == 'passport':
            required_terms = ['passport', 'date of birth', 'nationality', 'issue date']
            for term in required_terms:
                if term not in text_lower:
                    results['institutional_errors'].append({
                        'missing_term': term,
                        'expected_in': 'passport'
                    })
            
            if results['institutional_errors']:
                results['suspicious_indicators'].append(
                    f"⚠️ Missing {len(results['institutional_errors'])} expected passport terms"
                )
        
        return results


class EnhancedDocumentVerifier:
    """Main document verification class combining all analysis methods"""
    
    def __init__(self):
        self.forensics = DocumentForensics()
        self.date_analyzer = DateAnalyzer()
        self.spelling_analyzer = SpellingAnalyzer()
        self.current_date = datetime.now()
    
    def verify_document(self, file_path: str) -> Dict[str, Any]:
        """
        Comprehensive document verification
        Returns detailed verification results
        """
        results = {
            'file_info': self._get_file_info(file_path),
            'authenticity_score': 0,
            'risk_level': 'UNKNOWN',
            'metadata_analysis': {},
            'content_verification': {},
            'image_forensics': {},
            'ml_analysis': {},
            'recommendations': []
        }
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                results = self._verify_image(file_path, results)
            elif file_ext == '.pdf':
                results = self._verify_pdf(file_path, results)
            elif file_ext in ['.docx', '.doc']:
                results = self._verify_document_file(file_path, results)
            else:
                results['error'] = f"Unsupported file type: {file_ext}"
                return results
            
            # Calculate final authenticity score
            results['authenticity_score'] = self._calculate_authenticity_score(results)
            results['risk_level'] = self._determine_risk_level(results['authenticity_score'])
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['debug_info'] = traceback.format_exc()
        
        return results
    
    def _verify_image(self, file_path: str, results: Dict) -> Dict:
        """Verify image documents"""
        results['image_forensics'] = self.forensics.analyze_image_metadata(file_path)
        
        # Try to extract text using PIL (basic OCR-like analysis)
        try:
            img = Image.open(file_path)
            
            # Check image quality
            if img.size[0] < 200 or img.size[1] < 200:
                results['image_forensics']['suspicious_indicators'].append(
                    "⚠️ Image resolution too low for document verification"
                )
                results['image_forensics']['manipulation_score'] += 20
            
            # Check for common editing software artifacts
            for artifact in ['photoshop', 'gimp', 'paint', 'canva', 'pixlr']:
                if artifact in str(results['image_forensics'].get('metadata', {})).lower():
                    results['image_forensics']['suspicious_indicators'].append(
                        f"⚠️ Image created/edited with {artifact} - may have been tampered"
                    )
                    results['image_forensics']['manipulation_score'] += 30
        
        except Exception as e:
            results['image_forensics']['error'] = str(e)
        
        return results
    
    def _verify_pdf(self, file_path: str, file_results: Dict) -> Dict:
        """Verify PDF documents"""
        file_results['metadata_analysis'] = self.forensics.analyze_pdf_metadata(file_path)
        
        try:
            text_content = self._extract_pdf_text(file_path)
            file_results['content_verification'] = self._analyze_content(text_content)
        except Exception as e:
            file_results['content_verification']['error'] = str(e)
        
        return file_results
    
    def _verify_document_file(self, file_path: str, file_results: Dict) -> Dict:
        """Verify DOCX/DOC documents"""
        try:
            doc = Document(file_path)
            text_content = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            
            # Check document metadata
            file_results['metadata_analysis'] = {
                'suspicious_indicators': [],
                'manipulation_score': 0
            }
            
            if hasattr(doc, 'core_properties'):
                props = doc.core_properties
                if props.modified and props.created:
                    file_results['metadata_analysis']['edit_history'] = {
                        'created': str(props.created),
                        'modified': str(props.modified)
                    }
            
            file_results['content_verification'] = self._analyze_content(text_content)
        
        except Exception as e:
            file_results['metadata_analysis'] = {
                'error': str(e),
                'suspicious_indicators': [],
                'manipulation_score': 0
            }
        
        return file_results
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        text_content = []
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
        except:
            pass
        return '\n'.join(text_content)
    
    def _analyze_content(self, text_content: str) -> Dict[str, Any]:
        """Analyze document content"""
        results = {
            'expiry_validation': {},
            'spelling_grammar': {},
            'suspicious_indicators': []
        }
        
        if not text_content or len(text_content.strip()) < 10:
            results['suspicious_indicators'].append("⚠️ Document content is empty or too short")
            return results
        
        # Date analysis
        dates = self.date_analyzer.extract_dates(text_content)
        expiry_dates = self.date_analyzer.find_expiry_dates(text_content)
        results['expiry_validation'] = self.date_analyzer.validate_dates(dates, self.current_date)
        
        if expiry_dates:
            results['expiry_validation']['expiry_dates_found'] = expiry_dates
        
        # Spelling analysis
        results['spelling_grammar'].update(self.spelling_analyzer.check_spelling(text_content))
        results['spelling_grammar'].update(self.spelling_analyzer.check_institutional_terms(text_content))
        
        return results
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Extract file information"""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            return {
                'filename': os.path.basename(file_path),
                'size_bytes': os.path.getsize(file_path),
                'mime_type': mime_type or 'unknown',
                'file_hash': self._calculate_file_hash(file_path)[:16]  # First 16 chars of hash
            }
        except:
            return {
                'filename': os.path.basename(file_path),
                'error': 'Could not extract file info'
            }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return 'N/A'
    
    def _calculate_authenticity_score(self, results: Dict) -> float:
        """Calculate overall authenticity score based on all analyses"""
        score = 100.0
        
        # Deduct points based on findings
        
        # Metadata analysis
        metadata = results.get('metadata_analysis', {})
        manipulation_score = metadata.get('manipulation_score', 0)
        score -= min(manipulation_score, 30)
        
        # Content verification
        content = results.get('content_verification', {})
        
        # Deduct for expired documents
        expiry = content.get('expiry_validation', {})
        if expiry.get('validity_status') == 'expired':
            score -= 35
        elif expiry.get('validity_status') == 'suspicious_future_date':
            score -= 25
        
        # Deduct for spelling errors
        spelling = content.get('spelling_grammar', {})
        spelling_errors = len(spelling.get('spelling_errors', []))
        if spelling_errors > 0:
            score -= min(spelling_errors * 3, 15)
        
        inst_errors = len(spelling.get('institutional_errors', []))
        if inst_errors > 0:
            score -= min(inst_errors * 5, 20)
        
        # Deduct for suspicious indicators
        for section_name, section_data in results.items():
            if isinstance(section_data, dict):
                suspicious = section_data.get('suspicious_indicators', [])
                if suspicious:
                    score -= min(len(suspicious) * 3, 20)
        
        # Image forensics
        image = results.get('image_forensics', {})
        if image.get('manipulation_score', 0) > 50:
            score -= 25
        
        # Ensure score is within 0-100
        return max(0, min(100, score))
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on authenticity score"""
        if score >= 80:
            return "🟢 LOW RISK - Document appears authentic"
        elif score >= 60:
            return "🟡 MEDIUM RISK - Further verification recommended"
        elif score >= 40:
            return "🔴 HIGH RISK - Document is suspicious"
        else:
            return "🔴 CRITICAL - High probability of forgery/tampering"
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check validity
        expiry = results.get('content_verification', {}).get('expiry_validation', {})
        if expiry.get('validity_status') == 'expired':
            recommendations.append(
                "❌ Document is EXPIRED - Obtain a renewal or replacement document before visa application"
            )
        
        # Check for content issues
        spelling = results.get('content_verification', {}).get('spelling_grammar', {})
        if spelling.get('institutional_errors'):
            recommendations.append(
                "⚠️ Document contains institutional terminology errors - Verify authenticity with issuing authority"
            )
        
        # Check for image manipulation
        image = results.get('image_forensics', {})
        if image.get('manipulation_score', 0) > 30:
            recommendations.append(
                "🔍 Image shows signs of manipulation - Request original document or certified copy"
            )
        
        # Check metadata
        metadata = results.get('metadata_analysis', {})
        if metadata.get('manipulation_score', 0) > 20:
            recommendations.append(
                "📋 Metadata anomalies detected - Request document directly from issuing authority"
            )
        
        # If score is low
        if results['authenticity_score'] < 60:
            recommendations.append(
                "🛑 RECOMMENDED: Request additional verification or certified copies from official sources"
            )
        
        if not recommendations:
            recommendations.append(
                "✅ Document appears authentic - Proceed with visa application"
            )
        
        return recommendations
