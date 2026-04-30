"""
Clinical data models.

Strict Pydantic models representing the structured clinical summary.
No diagnosis fields — this system performs triage and structuring only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class Onset(str, Enum):
    SUDDEN = "sudden"
    GRADUAL = "gradual"


class Symptom(BaseModel):
    """A single reported symptom with optional clinical qualifiers."""

    name: str = Field(..., description="Name of the symptom")
    duration: Optional[str] = Field(
        None, description="How long the symptom has been present"
    )
    severity: Optional[Severity] = Field(
        None, description="Severity: mild, moderate, or severe"
    )
    onset: Optional[Onset] = Field(
        None, description="Onset type: sudden or gradual"
    )


class MedicalHistory(BaseModel):
    """Patient's relevant medical background."""

    conditions: List[str] = Field(
        default_factory=list, description="Pre-existing conditions"
    )
    medications: List[str] = Field(
        default_factory=list, description="Current medications"
    )
    allergies: List[str] = Field(
        default_factory=list, description="Known allergies"
    )


class VitalSigns(BaseModel):
    """Optional vital-sign readings extracted from conversation."""

    heart_rate: Optional[int] = Field(None, description="Heart rate in bpm")
    blood_pressure: Optional[str] = Field(
        None, description="Blood pressure as systolic/diastolic string, e.g. '120/80'"
    )
    temperature: Optional[float] = Field(
        None, description="Body temperature in °C"
    )
    oxygen_saturation: Optional[int] = Field(
        None, description="SpO2 percentage"
    )


class ClinicalSummary(BaseModel):
    """
    Structured clinical summary produced by the Structuring Agent.

    This model is intentionally free of diagnosis fields.
    It captures objective patient data extracted from conversation.
    """

    patient_id: Optional[str] = Field(
        None, description="Patient identifier if available"
    )
    chief_complaint: str = Field(
        ..., description="Primary reason the patient is seeking care"
    )
    symptoms: List[Symptom] = Field(
        default_factory=list, description="Reported symptoms"
    )
    medical_history: MedicalHistory = Field(
        default_factory=MedicalHistory,
        description="Patient medical history",
    )
    vital_signs: Optional[VitalSigns] = Field(
        None, description="Vital signs if mentioned"
    )
    summary_text: str = Field(
        ..., description="Concise narrative summary of the clinical picture"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the summary was generated",
    )
