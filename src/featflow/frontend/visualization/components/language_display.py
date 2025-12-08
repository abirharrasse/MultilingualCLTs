from dash import html
import math
from typing import Dict, Any, Optional

class LanguageDisplay:
    """Component for displaying multilingual feature analysis."""
    
    # Language mapping from full names to 2-letter codes
    LANGUAGE_MAPPING = {
        'fra_Latn': 'fr',
        'eng': 'en', 
        'deu_Latn': 'ge',
        'arb_Arab': 'ar',
        'cmn_Hani': 'ch'
    }
    
    # Color mapping for languages - lighter colors
    LANGUAGE_COLORS = {
        'en': '#86efac',  # light green
        'fr': '#93c5fd',  # light blue
        'ge': '#fdba74',  # light orange
        'ar': '#fca5a5',  # light red
        'ch': '#fde047'   # light yellow
    }
    
    @staticmethod
    def _calculate_entropy(distribution: Dict[str, float]) -> float:
        if not distribution:
            return 0
        
        values = [v for v in distribution.values() if v is not None and v > 0]
        if not values:
            return 0
            
        total = sum(values)
        if total == 0:
            return 0
        
        entropy = -sum((v/total) * math.log2(v/total) for v in values)
        return entropy
    
    @staticmethod
    def _normalize_language_keys(distribution: Dict[str, float]) -> Dict[str, float]:
        """Convert language keys to 2-letter codes, filtering out None values."""
        normalized = {}
        for lang, prob in distribution.items():
            if prob is not None:  # Filter out None values
                short_lang = LanguageDisplay.LANGUAGE_MAPPING.get(lang, lang)
                normalized[short_lang] = prob
        return normalized
    
    @staticmethod
    def create_language_bars(distribution: Dict[str, float], title: str, entropy: float) -> html.Div:
        """Create compact horizontal bar chart for language distribution."""
        if not distribution:
            return html.Div([
                html.Div(title, style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'color': '#374151',
                    'marginBottom': '3px'
                }),
                html.Div("No data", style={
                    'fontSize': '9px',
                    'color': '#6b7280',
                    'fontStyle': 'italic'
                })
            ])
        
        # Normalize and filter out None values
        normalized_dist = LanguageDisplay._normalize_language_keys(distribution)
        if not normalized_dist:
            return html.Div([
                html.Div(title, style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'color': '#374151',
                    'marginBottom': '3px'
                }),
                html.Div("No valid data", style={
                    'fontSize': '9px',
                    'color': '#6b7280',
                    'fontStyle': 'italic'
                })
            ])
        
        # Sort by probability, handling None values
        sorted_langs = sorted(
            [(k, v) for k, v in normalized_dist.items() if v is not None and v > 0], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if not sorted_langs:
            return html.Div([
                html.Div(title, style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'color': '#374151',
                    'marginBottom': '3px'
                }),
                html.Div("No valid data", style={
                    'fontSize': '9px',
                    'color': '#6b7280',
                    'fontStyle': 'italic'
                })
            ])
        
        # Color entropy based on threshold
        entropy_color = '#16a34a' if entropy > 0.8 else '#dc2626'
        entropy_bg = '#dcfce7' if entropy > 0.8 else '#fef2f2'
        
        bars = []
        for lang, prob in sorted_langs:
            color = LanguageDisplay.LANGUAGE_COLORS.get(lang, '#d1d5db')
            width_percent = prob * 100
            
            bars.append(
                html.Div([
                    html.Div([
                        html.Span(lang.upper(), style={
                            'fontSize': '9px',
                            'fontWeight': '500',
                            'color': '#374151',
                            'marginRight': '4px'
                        }),
                        html.Span(f"{prob:.2f}", style={
                            'fontSize': '8px',
                            'color': '#6b7280'
                        })
                    ], style={
                        'backgroundColor': color,
                        'height': '14px',
                        'width': f'{max(width_percent, 25)}%',
                        'display': 'flex',
                        'alignItems': 'center',
                        'paddingLeft': '4px',
                        'paddingRight': '2px',
                        'borderRadius': '2px',
                        'minWidth': '50px',
                        'overflow': 'hidden',
                        'whiteSpace': 'nowrap'
                    })
                ], style={
                    'marginBottom': '1px',
                    'display': 'flex',
                    'alignItems': 'center'
                })
            )
        
        return html.Div([
            html.Div([
                html.Span(f"{title}:", style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'color': '#374151',
                    'marginRight': '6px'
                }),
                html.Span(f"H={entropy:.2f}", style={
                    'fontSize': '9px',
                    'fontWeight': '600',
                    'color': entropy_color,
                    'backgroundColor': entropy_bg,
                    'padding': '1px 4px',
                    'borderRadius': '3px'
                })
            ], style={
                'marginBottom': '4px',
                'display': 'flex',
                'alignItems': 'center'
            }),
            html.Div(bars, style={
                'marginBottom': '2px'
            })
        ])
    
    @staticmethod
    def create_language_analysis(feature_config: Dict[str, Any]) -> Optional[html.Div]:
        """Create complete language analysis display for a feature."""
        lang_dist = feature_config.get('language_distribution')
        general_lang_dist = feature_config.get('general_language_distribution')
        
        if not lang_dist and not general_lang_dist:
            return None
        
        components = []
        
        if lang_dist:
            entropy_top = LanguageDisplay._calculate_entropy(lang_dist)
            components.append(
                LanguageDisplay.create_language_bars(
                    lang_dist, 
                    "Top Sequences", 
                    entropy_top
                )
            )
        
        if general_lang_dist:
            entropy_general = LanguageDisplay._calculate_entropy(general_lang_dist)
            components.append(
                LanguageDisplay.create_language_bars(
                    general_lang_dist,
                    "All Activations", 
                    entropy_general
                )
            )
        
        if not components:
            return None
        
        return html.Div([
            html.Div(components, style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',
                'gap': '8px'
            })
        ], style={
            'padding': '6px 8px',
            'backgroundColor': '#f3f4f6',
            'borderRadius': '4px',
            'marginBottom': '6px',
            'fontSize': '11px'
        })