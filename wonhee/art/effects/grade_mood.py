"""Soft pastel and matte moods."""
from ..grading import Grade, Finish, Look, register_look, Curve, s_curve

register_look(Look(
    "pastel-dream", "soft, glowing, candy, dreamy",
    "Lifted blacks, pastel saturation, pink-lavender shadows and mint-cream highlights, strong diffusion glow and hazy bloom.",
    "flowers, portraits, skies, interiors, light pictures.",
    "dark/contrasty pictures wash out; fine detail softened on purpose.",
    lambda p: Grade(exposure=0.3, lift=0.07, cap=0.96, shoulder=0.6, knee=0.6, sat=-0.12, vib=0.25, curve=Curve([(0, 0.05), (0.3, 0.36), (0.7, 0.75), (1, 0.98)]),
                    split=dict(sh=(330, 0.03), mid=(300, 0.01), hi=(170, 0.02)), skin=0.6),
    lambda p: Finish(grain=0.12, soften=0.55, bloom=(0.35, 26, 0.5, (1, 0.9, 1.0))),
    tags=["pastel", "soft", "dreamy", "light"]))

register_look(Look(
    "matte-pastel", "clean matte, calm, social-media",
    "Black point raised to a soft slate, muted pastel colour, slight cool-pink tint, minimal grain, no glow. Calmer and sharper than pastel-dream.",
    "product, interiors, portraits, landscapes, UI hero photos behind text.",
    "needs no punch; deep blacks are intentionally gone.",
    lambda p: Grade(lift=0.08, cap=0.97, sat=-0.18, vib=0.12, curve=s_curve(0.2), split=dict(sh=(285, 0.02), hi=(20, 0.012)), skin=0.6, shoulder=0.35),
    lambda p: Finish(grain=0.1),
    tags=["matte", "pastel", "calm", "clean"]))
