import cv2
import mediapipe as mp
import time
import pygame
from collections import Counter

# Initialize pygame mixer for sound effects
pygame.mixer.init()
countdown_sound = pygame.mixer.Sound("beep.wav")
win_sound = pygame.mixer.Sound("win.wav")

def detect_hand_shape(landmarks):
    """Determines if the shape is Rock, Paper, or Scissors based on hand landmarks."""

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    extended = sum(1 for finger in fingers if abs(finger.x - landmarks[5].x) > 0.1) # Count extended fingers
    
    if extended == 0:
        return "Rock"
    elif extended == 2:
        return "Scissors"
    elif extended > 2:
        return "Paper"
    else:
        return None
    
def determine_winner(gesture1, gesture2):
    """Determines round result based on RPS rules. Returns 1 or 2 for winning player or 'Tie'."""
    if gesture1 == gesture2:
        return "Tie"
    if (gesture1 == "Rock" and gesture2 == "Scissors") or \
       (gesture1 == "Scissors" and gesture2 == "Paper") or \
       (gesture1 == "Paper" and gesture2 == "Rock"):
        return 1
    else:
        return 2

def smooth_gesture(history):
    """Returns the most common gesture from history if enough consistent data exists."""
    if not history:
        return None
    count = Counter(history)
    gesture, freq = count.most_common(1)[0]
    # Need at least 3 out of 5 frames to count gesture as stable
    return gesture if freq >= 3 else None

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Game state variables
    score1, score2 = 0, 0
    last_smoothed = [None, None]
    round_start_time = None
    cooldown = False
    cooldown_start_time = None
    cooldown_duration = 2
    round_result = None
    required_hold_time = 2

    # Gesture smoothing buffers per player
    gesture_history = [[], []]
    history_length = 5

    # For beeping every 0.5s during countdown
    beep_intervals = [0.0, 0.67, 1.33]
    beep_timestamps_played = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert frame for processing
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Store raw detected gestures before smoothing and hand data
        raw_gestures = [None, None]
        player_hands = []

        if result.multi_hand_landmarks:
            # Draw landmarks and determine gesture for each hand
            for hand_landmarks in result.multi_hand_landmarks:
                # Calculate avg x-coordinate for player assignmnet
                avg_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
                gesture = detect_hand_shape(hand_landmarks.landmark)
                player_hands.append((avg_x, gesture, hand_landmarks))

            # Only executed if two hands are detected
            if len(player_hands) >= 2:
                # Sort by x-coordinate
                player_hands.sort(key=lambda x: x[0])
                raw_gestures = [player_hands[0][1], player_hands[1][1]]
                # Draw landmarks per hand
                for _, _, landmarks in player_hands:
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                # Assigns gesture to player1 and none to player2 if only one hand detected
                raw_gestures = [player_hands[0][1], None]
                mp_draw.draw_landmarks(frame, player_hands[0][2], mp_hands.HAND_CONNECTIONS)
        
        # Update gesture history for smoothing
        for i in range(2):
            if raw_gestures[i] is not None:
                gesture_history[i].append(raw_gestures[i])
                if len(gesture_history[i]) > history_length:
                    gesture_history[i].pop(0)
            else:
                # If no gestures, clear history for that player
                gesture_history[i] = []
        
        # Compute smoothed gestures
        smoothed_gestures = [smooth_gesture(gesture_history[0]), smooth_gesture(gesture_history[1])]

        # Show detected gestures and apply overlay
        overlay = frame.copy()
        overlay_top_left = (40, 15)
        overlay_bottom_right = (350, 115)
        cv2.rectangle(overlay, overlay_top_left, overlay_bottom_right, (50, 50, 50), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        player1_text = f"Player 1: {smoothed_gestures[0] if smoothed_gestures[0] else 'Waiting'}"
        player2_text = f"Player 2: {smoothed_gestures[1] if smoothed_gestures[1] else 'Waiting'}"
        cv2.putText(frame, player1_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, player2_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Score board
        overlay_top_left = (30, 400)
        overlay_bottom_right = (400, 440)
        cv2.rectangle(overlay, overlay_top_left, overlay_bottom_right, (50, 50, 50), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        score_text = f"Score  P1: {score1}  P2: {score2}"
        cv2.putText(frame, score_text, (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Round timing and cooldown logic
        current_time = time.time()

        if not cooldown:
            # Both players need valid gestures to start round timer
            if smoothed_gestures[0] is not None and smoothed_gestures[1] is not None:
                # If gestures are stable from last frame, start countdown
                if smoothed_gestures == last_smoothed:
                    if round_start_time is None:
                        round_start_time = current_time
                        beep_timestamps_played = []
                    else:
                        hold_time = current_time - round_start_time
                        
                        # Play beep at each 0.5s interval
                        for t in beep_intervals:
                            if hold_time >= t and t not in beep_timestamps_played:
                                if countdown_sound:
                                    countdown_sound.play()
                                beep_timestamps_played.append(t)

                        # Draw progress bar for hold timer
                        bar_width = 300
                        bar_height = 20
                        bar_x = 50
                        bar_y = 360
                        # Outline of progress bar
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_width, bar_y+bar_height), (255, 255, 255), 2)
                        progress = min(hold_time / required_hold_time, 1.0)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(progress * bar_width), bar_y+bar_height), (0, 0, 255), -1)
                        remaining = max(0, required_hold_time - hold_time)

                        # Display countdown timer on screen
                        cv2.putText(frame, f"Hold for {remaining:.1f}s", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                        
                        if hold_time >= required_hold_time:
                            # Determine winners once gestures have been stable for required time
                            outcome = determine_winner(smoothed_gestures[0], smoothed_gestures[1])
                            if outcome == "Tie":
                                round_result = "Tie Round!"
                            elif outcome == 1:
                                round_result = "Player 1 wins!"
                                score1 += 1
                            elif outcome == 2:
                                round_result = "Player 2 wins!"
                                score2 += 1
                            cooldown = True
                            cooldown_start_time = current_time
                else:
                    # Gestures changed from last frame, reset countdown
                    round_start_time = None
            else:
                # Not enough info to start a round
                round_start_time = None
            last_smoothed = smoothed_gestures.copy()
        else:
            # Display round outcome during cooldown
            if round_result:
                cv2.putText(frame, round_result, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if win_sound and (current_time - cooldown_start_time) < 0.1:
                    win_sound.play()
            # Check if cooldown is over
            if current_time - cooldown_start_time > cooldown_duration:
                cooldown = False
                round_start_time = None
                last_smoothed = [None, None]
                round_result = None
                gesture_history = [[], []]

        cv2.imshow("Two-Player RPS - Gesture Detection", frame)
        # Exit by clicking the "esc" button
        if cv2.waitKey(1) & 0xFF == 27:
            break
      
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()