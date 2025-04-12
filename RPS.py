import cv2
import mediapipe as mp
import time
import pygame
import numpy as np

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


# Detects hand gesture
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


# Determines winner
def determine_winner(gesture1, gesture2):
    if gesture1 == gesture2:
        return "Tie"
    if (gesture1 == "Rock" and gesture2 == "Scissors") or \
       (gesture1 == "Scissors" and gesture2 == "Paper") or \
       (gesture1 == "Paper" and gesture2 == "Rock"):
        return 1
    else:
        return 2


# Applies weighted smoothing
def advanced_smooth_gesture(history):
    if not history:
        return None

    weights = {}
    for i, gesture in enumerate(history):
        weight = i + 1
        weights[gesture] = weights.get(gesture, 0) + weight

    return max(weights, key=weights.get)


# Game logic
class RPSGame:
    def __init__(self):
        self.icons = {
            "Rock": cv2.imread("rock.png", cv2.IMREAD_UNCHANGED),
            "Paper": cv2.imread("paper.png", cv2.IMREAD_UNCHANGED),
            "Scissors": cv2.imread("scissors.png", cv2.IMREAD_UNCHANGED),
            "Waiting": cv2.imread("waiting.png", cv2.IMREAD_UNCHANGED)
        }

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        pygame.mixer.init()
        self.countdown_sound = pygame.mixer.Sound("beep.wav")
        self.win_sound = pygame.mixer.Sound("win.wav")

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

    # Draw gesture icon
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

    # Get processed frame
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        return frame, result

    # Draw particles
    def update_particles(self, frame):
        for particle in self.particles[:]:
            particle.update()
            cv2.circle(frame, (int(particle.x), int(particle.y)), 3, (0, 255, 255), -1)
            if particle.life <= 0:
                self.particles.remove(particle)

    # Add confetti
    def add_particles(self, frame):
        for _ in range(30):
            x = np.random.randint(0, frame.shape[1])
            y = 0
            self.particles.append(Particle(x, y))

    # Reset everything
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

    # Run main loop
    def run(self):
        prev_time = time.time()

        while self.cap.isOpened():
            frame, result = self.process_frame()
            if frame is None:
                break

            raw_gestures = [None, None]
            player_hands = []

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

            # Draw gesture icons
            self.overlay_icon(frame, smoothed_gestures[0] or "Waiting", (200, 5))
            self.overlay_icon(frame, smoothed_gestures[1] or "Waiting", (500, 5))

            # Label players
            cv2.putText(frame, "Player 1:", (30, 50), self.font, 1, (0, 0, 0), 2)
            cv2.putText(frame, "Player 2:", (325, 50), self.font, 1, (0, 0, 0), 2)

            # Scoreboard
            overlay = frame.copy()
            overlay_top_left = (30, 400)
            overlay_bottom_right = (400, 440)
            cv2.rectangle(overlay, overlay_top_left, overlay_bottom_right, (50, 50, 50), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            score_text = f"Score  P1: {self.score1}  P2: {self.score2}"
            cv2.putText(frame, score_text, (40, 430), self.font, 1, (255, 255, 0), 2)

            # Round logic
            current_time = time.time()

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
                                    if self.countdown_sound:
                                        self.countdown_sound.play()
                                    self.beep_timestamps_played.append(t)

                            bar_width = 300
                            bar_height = 20
                            bar_x = 50
                            bar_y = 360
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

                            progress = min(hold_time / self.required_hold_time, 1.0)
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(progress * bar_width), bar_y + bar_height), (0, 0, 255), -1)

                            remaining = max(0, self.required_hold_time - hold_time)
                            cv2.putText(frame, f"Hold for {remaining:.1f}s", (bar_x, bar_y - 10), self.font, 0.8, (0, 0, 255), 2)

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

            # Animate particles
            self.update_particles(frame)

            cv2.imshow("Two-Player RPS - Gesture Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == ord('r'):
                self.reset_game()

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    game = RPSGame()
    game.run()


if __name__ == "__main__":
    main()