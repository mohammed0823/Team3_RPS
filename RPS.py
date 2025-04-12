import cv2
import mediapipe as mp
import time
import pygame
import numpy as np
import math

# Particle animation
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(1, 3)
        self.life = 50

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

# Gesture logic
def detect_hand_shape(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    extended = sum(1 for finger in fingers if abs(finger.x - landmarks[5].x) > 0.1)

    if extended == 0:
        return "Rock"
    elif extended == 2:
        return "Scissors"
    elif extended > 2:
        return "Paper"
    else:
        return None

def determine_winner(gesture1, gesture2):
    if gesture1 == gesture2:
        return "Tie"
    if (gesture1 == "Rock" and gesture2 == "Scissors") or \
       (gesture1 == "Scissors" and gesture2 == "Paper") or \
       (gesture1 == "Paper" and gesture2 == "Rock"):
        return 1
    else:
        return 2

def advanced_smooth_gesture(history):
    if not history:
        return None
    weights = {}
    for i, gesture in enumerate(history):
        weight = i + 1
        weights[gesture] = weights.get(gesture, 0) + weight
    return max(weights, key=weights.get)

# Game class
class RPSGame:
    def __init__(self):
        # Load icons
        self.icons = {
            "Rock": cv2.imread("rock.png", cv2.IMREAD_UNCHANGED),
            "Paper": cv2.imread("paper.png", cv2.IMREAD_UNCHANGED),
            "Scissors": cv2.imread("scissors.png", cv2.IMREAD_UNCHANGED),
            "Waiting": cv2.imread("waiting.png", cv2.IMREAD_UNCHANGED)
        }

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        # Sound setup
        pygame.mixer.init()
        self.countdown_sound = pygame.mixer.Sound("beep.wav")
        self.win_sound = pygame.mixer.Sound("win.wav")

        # Game state
        self.score1, self.score2 = 0, 0
        self.last_smoothed = [None, None]
        self.round_start_time = None
        self.cooldown = False
        self.cooldown_start_time = None
        self.cooldown_duration = 2
        self.round_result = None
        self.required_hold_time = 2

        self.gesture_history = [[], []]
        self.history_length = 5
        self.beep_intervals = [0.0, 0.67, 1.33]
        self.beep_timestamps_played = []
        self.particles = []

        self.confidence_threshold = 0.8
        self.font = cv2.FONT_HERSHEY_TRIPLEX

        # Game mode selection
        print("Select Game Mode:")
        print("1: Regular Mode")
        print("2: Best-of-N Mode")
        print("3: Timed Mode")
        mode_input = input("Enter 1, 2, or 3: ").strip()

        if mode_input == "2":
            self.mode = "best_of"
            best_of_val = input("Enter an odd number of rounds (e.g., 3, 5, 7): ").strip()
            self.best_of = int(best_of_val)
            self.win_threshold = math.ceil(self.best_of / 2)

        elif mode_input == "3":
            self.mode = "timed"
            timed_val = input("Enter match duration in seconds (e.g., 60): ").strip()
            self.match_duration = int(timed_val)

        else:
            self.mode = "regular"

        self.match_start_time = None

    # Countdown before match
    def display_countdown(self, seconds=3):
        for i in range(seconds, 0, -1):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, str(i), (270, 250), self.font, 3, (0, 255, 0), 5)
            cv2.imshow("Countdown", frame)
            cv2.setWindowProperty("Countdown", cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1000)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Go!", (240, 250), self.font, 3, (0, 255, 0), 5)
        cv2.imshow("Countdown", frame)
        cv2.waitKey(1000)
        cv2.destroyWindow("Countdown")

    # Display winner at end
    def display_victory_screen(self, message, duration=3000):
        start_time = time.time()
        while (time.time() - start_time) * 1000 < duration:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, message, (50, 250), self.font, 2, (0, 0, 255), 3)
            cv2.imshow("Victory", frame)
            cv2.setWindowProperty("Victory", cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
        cv2.destroyWindow("Victory")

    # Overlay icons
    def overlay_icon(self, frame, gesture, position):
        icon = self.icons.get(gesture, self.icons["Waiting"])
        if icon is None:
            return

        icon = cv2.resize(icon, (80, 80))
        x, y = position
        h, w = icon.shape[:2]

        if icon.shape[2] == 4:
            roi = frame[y:y+h, x:x+w]
            icon_rgb = icon[:, :, :3]
            alpha = icon[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha

            for c in range(3):
                roi[:, :, c] = (alpha * icon_rgb[:, :, c] + alpha_inv * roi[:, :, c])
            frame[y:y+h, x:x+w] = roi
        else:
            frame[y:y+h, x:x+w] = icon

    # Frame preprocessing
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        return frame, result

    # Confetti logic
    def update_particles(self, frame):
        for particle in self.particles[:]:
            particle.update()
            cv2.circle(frame, (int(particle.x), int(particle.y)), 3, (0, 255, 255), -1)
            if particle.life <= 0:
                self.particles.remove(particle)

    def add_particles(self, frame):
        for _ in range(30):
            x = np.random.randint(0, frame.shape[1])
            y = 0
            self.particles.append(Particle(x, y))

    # Reset game state
    def reset_game(self):
        self.score1 = 0
        self.score2 = 0
        self.last_smoothed = [None, None]
        self.round_start_time = None
        self.cooldown = False
        self.cooldown_start_time = None
        self.round_result = None
        self.gesture_history = [[], []]
        self.beep_timestamps_played = []
        self.particles = []
        self.match_start_time = None

    # Main game loop
    def run(self):
        self.display_countdown(seconds=3)

        if self.mode == "timed":
            self.match_start_time = time.time()

        prev_time = time.time()

        while self.cap.isOpened():
            frame, result = self.process_frame()
            if frame is None:
                break

            raw_gestures = [None, None]
            player_hands = []

            # Detect and filter hands
            if result.multi_hand_landmarks and result.multi_handedness:
                for i, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
                    confidence = handedness.classification[0].score
                    if confidence < self.confidence_threshold:
                        continue

                    avg_x = sum(lm.x for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                    gesture = detect_hand_shape(hand_landmarks.landmark)
                    player_hands.append((avg_x, gesture, hand_landmarks))

                if len(player_hands) >= 2:
                    player_hands.sort(key=lambda x: x[0])
                    raw_gestures = [player_hands[0][1], player_hands[1][1]]

                    for _, _, landmarks in player_hands:
                        self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

                elif len(player_hands) == 1:
                    raw_gestures = [player_hands[0][1], None]
                    self.mp_draw.draw_landmarks(frame, player_hands[0][2], self.mp_hands.HAND_CONNECTIONS)

            # Smooth gestures
            for i in range(2):
                if raw_gestures[i] is not None:
                    self.gesture_history[i].append(raw_gestures[i])
                    if len(self.gesture_history[i]) > self.history_length:
                        self.gesture_history[i].pop(0)
                else:
                    self.gesture_history[i] = []

            smoothed_gestures = [
                advanced_smooth_gesture(self.gesture_history[0]),
                advanced_smooth_gesture(self.gesture_history[1])
            ]

            # Show icons and labels
            self.overlay_icon(frame, smoothed_gestures[0] or "Waiting", (200, 5))
            self.overlay_icon(frame, smoothed_gestures[1] or "Waiting", (500, 5))
            cv2.putText(frame, "Player 1:", (30, 50), self.font, 1, (0, 0, 0), 2)
            cv2.putText(frame, "Player 2:", (325, 50), self.font, 1, (0, 0, 0), 2)

            # Show scoreboard
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 400), (375, 440), (50, 50, 50), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, f"Score  P1: {self.score1}  P2: {self.score2}", (15, 430), self.font, 1, (255, 255, 0), 2)

            # Timed mode check
            current_time = time.time()

            if self.mode == "timed" and self.match_start_time:
                elapsed = current_time - self.match_start_time
                remaining = max(0, self.match_duration - elapsed)
                cv2.putText(frame, f"Time Left: {int(remaining)}s", (380, 430), self.font, 1, (0, 255, 255), 2)

                if remaining <= 0:
                    if self.score1 == self.score2:
                        self.display_victory_screen("Tie Match!")
                    elif self.score1 > self.score2:
                        self.display_victory_screen("Player 1 Wins!")
                    else:
                        self.display_victory_screen("Player 2 Wins!")
                    break

            # Round handling
            if not self.cooldown:
                if smoothed_gestures[0] and smoothed_gestures[1]:
                    if smoothed_gestures == self.last_smoothed:
                        if self.round_start_time is None:
                            self.round_start_time = current_time
                            self.beep_timestamps_played = []
                        else:
                            hold_time = current_time - self.round_start_time

                            for t in self.beep_intervals:
                                if hold_time >= t and t not in self.beep_timestamps_played:
                                    self.countdown_sound.play()
                                    self.beep_timestamps_played.append(t)

                            bar_x, bar_y = 50, 360
                            bar_width, bar_height = 300, 20
                            progress = min(hold_time / self.required_hold_time, 1.0)

                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(progress * bar_width), bar_y + bar_height), (0, 0, 255), -1)
                            cv2.putText(frame, f"Hold for {max(0, self.required_hold_time - hold_time):.1f}s", (bar_x, bar_y - 10), self.font, 0.8, (0, 0, 255), 2)

                            if hold_time >= self.required_hold_time:
                                outcome = determine_winner(smoothed_gestures[0], smoothed_gestures[1])
                                if outcome == "Tie":
                                    self.round_result = "Tie Round!"
                                elif outcome == 1:
                                    self.round_result = "Player 1 wins!"
                                    self.score1 += 1
                                elif outcome == 2:
                                    self.round_result = "Player 2 wins!"
                                    self.score2 += 1

                                self.cooldown = True
                                self.cooldown_start_time = current_time
                    else:
                        self.round_start_time = None
                else:
                    self.round_start_time = None

                self.last_smoothed = smoothed_gestures.copy()

            else:
                if self.round_result:
                    cv2.putText(frame, self.round_result, (50, 350), self.font, 1.2, (0, 0, 255), 3)

                    if self.win_sound and (current_time - self.cooldown_start_time) < 0.1:
                        self.win_sound.play()

                    if (current_time - self.cooldown_start_time) < 0.2:
                        self.add_particles(frame)

                if current_time - self.cooldown_start_time > self.cooldown_duration:
                    self.cooldown = False
                    self.round_start_time = None
                    self.last_smoothed = [None, None]
                    self.round_result = None
                    self.gesture_history = [[], []]

            # Best-of-N mode check
            if self.mode == "best_of":
                if self.score1 >= self.win_threshold or self.score2 >= self.win_threshold:
                    if self.score1 == self.score2:
                        final = "Tie Match!"
                    elif self.score1 > self.score2:
                        final = "Player 1 Wins!"
                    else:
                        final = "Player 2 Wins!"
                    self.display_victory_screen(final)
                    break

            # Update particles and draw frame
            self.update_particles(frame)
            cv2.imshow("Two-Player RPS - Gesture Detection", frame)
            cv2.setWindowProperty("Two-Player RPS - Gesture Detection", cv2.WND_PROP_TOPMOST, 1)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('r'):
                self.reset_game()

        self.cap.release()
        cv2.destroyAllWindows()

# -------- Entry Point -------- #
def main():
    game = RPSGame()
    game.run()

if __name__ == "__main__":
    main()
