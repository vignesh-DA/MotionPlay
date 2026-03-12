"""
MODULE 5: GAME CONTROL INTERFACE
=================================

Purpose:
    Convert gesture labels into keyboard commands and send them to the game
    emulator with low latency, managing command timing to prevent repeated
    triggers and ensure responsive gameplay.

Responsibilities:
    - Maintain debounce timer to prevent rapid-fire commands
    - Execute keyboard commands via pyautogui
    - Track command history for statistics
    - Optional target window verification (BlueStacks)
    - Logging and debug information

Input:  Gesture label + confidence from Module 4
Output: Keyboard event to game emulator (via OS)

Author:     Capstone Project
Date:       2025
Version:    1.0
"""

import pyautogui
import time
from typing import Dict, Tuple, Optional, List
from collections import deque
from enum import Enum


class CommandType(Enum):
    """Enum for keyboard command types."""
    JUMP = "jump"
    SLIDE = "slide"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"


class GameControlInterface:
    """
    Manages keyboard command execution for game control.
    
    Attributes:
        command_cooldown (float): Minimum time (sec) between commands
        last_command_time (float): Timestamp of last command
        command_history (deque): Recent command log
        command_count (dict): Count of each command type
        min_confidence_threshold (float): Minimum confidence to execute
        log_commands (bool): Whether to log commands
    """
    
    # Keyboard key mapping for each gesture/action
    GESTURE_TO_KEY_MAP = {
        "jump": "up",
        "slide": "down",
        "move_right": "right",
        "move_left": "left",
    }
    
    def __init__(self, 
                 command_cooldown: float = 0.5,
                 min_confidence_threshold: float = 0.0,
                 log_commands: bool = True,
                 history_size: int = 100):
        """
        Initialize game control interface.
        
        Args:
            command_cooldown (float): Seconds between allowed commands (default 0.5)
                                     Typical game action duration: 300-500ms
            min_confidence_threshold (float): Reject gestures below threshold (0-1)
                                            Default 0.0 = accept all
            log_commands (bool): Enable command history logging
            history_size (int): Size of command history buffer
        """
        self.command_cooldown = command_cooldown
        self.last_command_time = 0.0
        self.min_confidence_threshold = min_confidence_threshold
        self.log_commands = log_commands
        
        # Command history tracking
        self.command_history = deque(maxlen=history_size)
        self.command_count = {cmd.value: 0 for cmd in CommandType}
        
        # Statistics
        self.total_commands_issued = 0
        self.total_commands_rejected = 0
        self.total_gestures_processed = 0
        
        print("✓ Game control interface initialized")
        print(f"  Command cooldown: {command_cooldown} seconds")
        print(f"  Confidence threshold: {min_confidence_threshold}")
        print(f"  Command history size: {history_size}")
    
    def check_debounce(self) -> bool:
        """
        Check if sufficient time has elapsed since last command (debouncing).
        
        Purpose:
            Prevent rapid repeated commands from jitter/noise in gesture detection.
            Ensures each gesture is completed before accepting next gesture.
        
        Mathematical Model:
            if (current_time - last_command_time) > cooldown:
                allowed = True
            else:
                allowed = False  # Still in cooldown period
        
        Returns:
            bool: True if command is allowed (outside cooldown), False otherwise
        
        Time Complexity: O(1)
        """
        current_time = time.time()
        elapsed = current_time - self.last_command_time
        
        is_allowed = elapsed > self.command_cooldown
        
        return is_allowed
    
    def execute_keyboard_command(self, key_name: str) -> bool:
        """
        Execute keyboard command via pyautogui.
        
        Implementation:
            Uses pyautogui.press(key) which:
            1. Simulates key press (OS-level keyboard event)
            2. Holds for ~50ms (typical key press duration)
            3. Simulates key release
            4. Direct to OS (no window focus required)
        
        Supported Keys:
            'up', 'down', 'left', 'right' (arrow keys)
            Works with any OS that supports pyautogui
        
        Safety:
            - pyautogui has built-in fail-safe (move mouse to corner to abort)
            - Keys are validated before execution
            
        Args:
            key_name (str): Key name (e.g., 'up', 'down', 'left', 'right')
        
        Returns:
            bool: True if command executed successfully, False otherwise
        
        Time Complexity: O(1)
        Expected Duration: 50-100 ms (key press duration)
        """
        try:
            # Validate key name
            valid_keys = ['up', 'down', 'left', 'right']
            if key_name not in valid_keys:
                return False
            
            # Execute keyboard press
            pyautogui.press(key_name)
            
            return True
        
        except Exception as e:
            print(f"✗ Error executing keyboard command '{key_name}': {str(e)}")
            return False
    
    def should_accept_gesture(self, confidence: float) -> bool:
        """
        Determine if gesture should be accepted based on confidence threshold.
        
        Logic:
            if confidence >= min_confidence_threshold:
                accept = True
            else:
                accept = False (too low confidence)
        
        Args:
            confidence (float): Confidence score [0.0, 1.0]
        
        Returns:
            bool: True if gesture meets threshold, False otherwise
        
        Time Complexity: O(1)
        """
        return confidence >= self.min_confidence_threshold
    
    def execute_gesture_command(self, action: Optional[str], confidence: float) -> bool:
        """
        Execute command for a gesture.
        
        Full Pipeline:
            1. Check if confidence meets threshold
            2. Check if debounce timer allows new command
            3. Map action to keyboard key
            4. Execute keyboard press
            5. Update cooldown timer
            6. Log command
            7. Update statistics
        
        Args:
            action (str): Action type ('jump', 'slide', 'move_left', 'move_right')
                         None if gesture has no action
            confidence (float): Confidence score [0.0, 1.0]
        
        Returns:
            bool: True if command was executed, False otherwise
        
        Time Complexity: O(1)
        Expected duration: ~50-100 ms for actual key press
        """
        self.total_gestures_processed += 1
        
        # Step 1: Check confidence threshold
        if not self.should_accept_gesture(confidence):
            self.total_commands_rejected += 1
            return False
        
        # Step 2: Check if action is valid (not None)
        if action is None:
            return False
        
       # Step 3: Check debounce
        if not self.check_debounce():
            self.total_commands_rejected += 1
            return False
        
        # Step 4: Map action to key
        key_name = self.GESTURE_TO_KEY_MAP.get(action, None)
        if key_name is None:
            return False
        
        # Step 5: Execute keyboard command
        success = self.execute_keyboard_command(key_name)
        
        if success:
            # Step 6: Update cooldown timer
            self.last_command_time = time.time()
            
            # Step 7: Log command
            if self.log_commands:
                self._log_command(action, key_name, confidence)
            
            # Step 8: Update statistics
            self.total_commands_issued += 1
            self.command_count[action] += 1
            
            return True
        
        return False
    
    def _log_command(self, action: str, key: str, confidence: float) -> None:
        """
        Log command to history for debugging and statistics.
        
        Args:
            action (str): Action type
            key (str): Keyboard key pressed
            confidence (float): Confidence score
        """
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'action': action,
            'key': key,
            'confidence': confidence,
            'elapsed_since_prev': timestamp - self.last_command_time
        }
        
        self.command_history.append(log_entry)
    
    def get_command_statistics(self) -> dict:
        """
        Get command execution statistics.
        
        Returns:
            dict: Statistics including:
                - 'total_gestures_processed': Total gestures seen
                - 'total_commands_issued': Successful commands
                - 'total_commands_rejected': Rejected gestures
                - 'command_counts': Dict of action → count
                - 'execution_rate': Percentage of gestures executed
                - 'cooldown_value': Current cooldown setting
        """
        total_processed = self.total_gestures_processed
        
        if total_processed == 0:
            execution_rate = 0.0
        else:
            execution_rate = 100 * self.total_commands_issued / total_processed
        
        return {
            'total_gestures_processed': total_processed,
            'total_commands_issued': self.total_commands_issued,
            'total_commands_rejected': self.total_commands_rejected,
            'command_counts': dict(self.command_count),
            'execution_rate': execution_rate,
            'cooldown_value': self.command_cooldown,
            'min_confidence_threshold': self.min_confidence_threshold,
        }
    
    def get_recent_commands(self, count: int = 10) -> List[dict]:
        """
        Get recent commands from history.
        
        Args:
            count (int): Number of recent commands to return
        
        Returns:
            List[dict]: List of command log entries (most recent last)
        """
        return list(self.command_history)[-count:]
    
    def set_cooldown(self, cooldown: float) -> None:
        """
        Update command cooldown value.
        
        Args:
            cooldown (float): New cooldown in seconds
        """
        self.command_cooldown = cooldown
        print(f"✓ Command cooldown updated to {cooldown} seconds")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold for gesture acceptance.
        
        Args:
            threshold (float): New threshold [0.0, 1.0]
        """
        self.min_confidence_threshold = threshold
        print(f"✓ Confidence threshold updated to {threshold}")
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.total_commands_issued = 0
        self.total_commands_rejected = 0
        self.total_gestures_processed = 0
        self.command_count = {cmd.value: 0 for cmd in CommandType}
        self.command_history.clear()
        print("✓ Statistics reset")


if __name__ == "__main__":
    """
    Test Module 5 independently.
    
    WARNING: This will actually send keyboard commands!
    Use with caution...
    """
    print("=" * 60)
    print("MODULE 5: GAME CONTROL INTERFACE TEST")
    print("=" * 60)
    print("\nWARNING: This test will send actual keyboard commands!")
    print("Starting in 3 seconds... press Ctrl+C to abort\n")
    
    try:
        import time as time_module
        time_module.sleep(3)
        
        # Initialize game control
        controller = GameControlInterface(
            command_cooldown=0.5,
            min_confidence_threshold=0.0,
            log_commands=True
        )
        
        # Test gesture→action→key mapping
        test_gestures = [
            ("jump", 0.95),      # OPEN_PALM
            ("slide", 0.95),     # CLOSED_FIST  
            ("move_right", 0.85), # TWO_FINGERS
            ("move_left", 0.86),  # THREE_FINGERS
            (None, 0.50),        # UNDEFINED (no action)
        ]
        
        print("Testing gesture execution with cooldown=0.5s:")
        print("-" * 60)
        
        for action, confidence in test_gestures:
            print(f"\nGesture: {action}, Confidence: {confidence:.2f}")
            
            # Attempt to execute
            success = controller.execute_gesture_command(action, confidence)
            
            if success:
                print("  ✓ Command executed")
                key = controller.GESTURE_TO_KEY_MAP.get(action, "?")
                print(f"    Key pressed: {key}")
            else:
                print("  ✗ Command rejected or failed")
            
            # Wait before next command (for visibility)
            print("  Waiting 1 second before next command...")
            time_module.sleep(1.0)
        
        # Print statistics
        print("\n" + "-" * 60)
        print("Execution Statistics:")
        print("-" * 60)
        
        stats = controller.get_statistics()
        print(f"Total gestures processed: {stats['total_gestures_processed']}")
        print(f"Total commands issued: {stats['total_commands_issued']}")
        print(f"Total commands rejected: {stats['total_commands_rejected']}")
        print(f"Execution rate: {stats['execution_rate']:.1f}%")
        print(f"\nCommand breakdown:")
        for action, count in stats['command_counts'].items():
            if count > 0:
                print(f"  {action}: {count}")
        
        # Print command history
        print(f"\n" + "-" * 60)
        print("Recent Commands:")
        print("-" * 60)
        
        for entry in controller.get_recent_commands(5):
            print(f"Action: {entry['action']:<12s} " +
                  f"Key: {entry['key']:<6s} " +
                  f"Conf: {entry['confidence']:.2f} " +
                  f"Elapsed: {entry['elapsed_since_prev']:.2f}s")
        
        print("\n" + "=" * 60)
        print("✓ TEST COMPLETE")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n✗ Test aborted by user")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
