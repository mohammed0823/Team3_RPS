# Two-Player Rock Paper Scissors – AI Gesture Edition

A webcam-based Rock Paper Scissors game built in 8 hours for a hackathon. Players use real hand gestures, detected via AI, to battle it out on screen.

## How It Works
- Uses **MediaPipe** for real-time hand tracking.
- Recognizes Rock, Paper, or Scissors gestures.
- Displays icons, animations, and sound effects.
- Supports:
  - Regular mode (endless)
  - Best-of-N mode
  - Timed mode (play against the clock)

## Features
- Gesture smoothing for better accuracy
- Confidence filtering to reduce misfires
- Countdown and victory screens
- Confetti particle animation on win
- Sound effects and score tracking
- Game restart with `R` key

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pygame

Install dependencies:
```bash
pip install opencv-python mediapi pe numpy pygame
```

## Run It
```bash
python RPS.py
```
Then follow the on-screen prompts to pick a game mode and play!
