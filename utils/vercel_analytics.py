#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vercel Web Analytics integration for Gradio applications.

This module provides utilities to integrate Vercel Web Analytics
into Gradio-based web interfaces.
"""

import os
from typing import Optional


def get_analytics_script() -> Optional[str]:
    """
    Get the Vercel Analytics injection script for Gradio.
    
    The script uses the generic analytics approach from Vercel docs,
    which works for any framework including Python/Gradio apps.
    
    Returns:
        JavaScript code to inject analytics, or None if analytics is disabled.
    """
    # Check if analytics is enabled via environment variable
    analytics_enabled = os.getenv("VERCEL_ANALYTICS_ENABLED", "false").lower() == "true"
    
    if not analytics_enabled:
        return None
    
    # Vercel Analytics script injection (as per official docs for generic frameworks)
    # This approach works for any web framework, including Python/Gradio
    analytics_js = """
    window.va = window.va || function () { 
        (window.vaq = window.vaq || []).push(arguments); 
    };
    """
    
    return analytics_js


def get_analytics_head() -> Optional[str]:
    """
    Get the Vercel Analytics HTML head content for Gradio.
    
    Returns:
        HTML to inject into the document head, or None if analytics is disabled.
    """
    # Check if analytics is enabled via environment variable
    analytics_enabled = os.getenv("VERCEL_ANALYTICS_ENABLED", "false").lower() == "true"
    
    if not analytics_enabled:
        return None
    
    # Get the analytics ID from environment variable
    # After deploying on Vercel, this will be set automatically
    # For local development, users can set it manually
    analytics_id = os.getenv("VERCEL_ANALYTICS_ID", "")
    
    if analytics_id:
        # If a custom analytics ID is provided, use it
        script_path = f"/_vercel/insights/script.js?id={analytics_id}"
    else:
        # Use the default Vercel path - this will be set automatically on Vercel deployments
        script_path = "/_vercel/insights/script.js"
    
    head_html = f'<script defer src="{script_path}"></script>'
    
    return head_html


def get_launch_kwargs(existing_kwargs: Optional[dict] = None) -> dict:
    """
    Get launch kwargs for Gradio with Vercel Analytics integrated.
    
    Args:
        existing_kwargs: Existing launch arguments to merge with analytics config.
    
    Returns:
        Dictionary of launch arguments including analytics configuration.
    """
    kwargs = existing_kwargs.copy() if existing_kwargs else {}
    
    # Get analytics script and head content
    analytics_js = get_analytics_script()
    analytics_head = get_analytics_head()
    
    # Only add analytics if it's enabled
    if analytics_js or analytics_head:
        # Merge with existing js parameter
        if analytics_js:
            existing_js = kwargs.get("js", "")
            if existing_js:
                kwargs["js"] = f"{existing_js}\n{analytics_js}"
            else:
                kwargs["js"] = analytics_js
        
        # Merge with existing head parameter
        if analytics_head:
            existing_head = kwargs.get("head", "")
            if existing_head:
                kwargs["head"] = f"{existing_head}\n{analytics_head}"
            else:
                kwargs["head"] = analytics_head
    
    return kwargs


def is_analytics_enabled() -> bool:
    """
    Check if Vercel Analytics is enabled.
    
    Returns:
        True if analytics is enabled, False otherwise.
    """
    return os.getenv("VERCEL_ANALYTICS_ENABLED", "false").lower() == "true"


def get_analytics_status() -> dict:
    """
    Get the current analytics configuration status.
    
    Returns:
        Dictionary with analytics status information.
    """
    enabled = is_analytics_enabled()
    analytics_id = os.getenv("VERCEL_ANALYTICS_ID", "")
    
    return {
        "enabled": enabled,
        "analytics_id": analytics_id if analytics_id else "auto (from Vercel deployment)",
        "message": "Vercel Analytics is " + ("enabled" if enabled else "disabled"),
    }
