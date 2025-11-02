#!/usr/bin/env python3
"""
Webcam Viewport - Display live webcam feed
"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# Try to import OpenCV, gracefully handle if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  opencv-python not installed. Webcam viewport will be disabled.")
    print("   To enable: pip install opencv-python")


class WebcamViewport(QWidget):
    """
    Viewport for displaying live webcam feed
    
    Features:
    - Live video from default webcam (camera index 0)
    - Automatic frame rate (30 FPS)
    - Auto-resize to viewport dimensions
    - Error handling for missing/busy camera
    """
    
    def __init__(self, camera_index: int = 0, fps: int = 30):
        """
        Initialize webcam viewport
        
        Args:
            camera_index: Camera device index (0 = default)
            fps: Target frame rate
        """
        super().__init__()
        
        self.camera_index = camera_index
        self.fps = fps
        self.capture = None
        self.timer = None
        
        self._init_ui()
        self._init_camera()
        
    def _init_ui(self):
        """Initialize the user interface"""
        from PyQt5.QtWidgets import QSizePolicy
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("📷 Webcam")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                background-color: #2d2d30;
                padding: 5px;
            }
        """)
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(title)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 1px solid #555;
            }
        """)
        # Set size policy to prevent growing beyond container
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(100, 100)  # Prevent collapsing
        # Don't use setScaledContents - we handle scaling manually in _update_frame
        layout.addWidget(self.video_label, stretch=1)
        
        # Status label
        self.status_label = QLabel("Initializing camera...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 10px;
                padding: 2px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Set size policy to prevent widget from growing
        from PyQt5.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
            }
        """)
    
    def _init_camera(self):
        """Initialize camera capture"""
        if not CV2_AVAILABLE:
            self.status_label.setText("❌ OpenCV not installed")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 10px;
                    padding: 2px;
                }
            """)
            self.video_label.setText("pip install opencv-python")
            self.video_label.setStyleSheet("""
                QLabel {
                    background-color: black;
                    border: 1px solid #555;
                    color: #888;
                    padding: 20px;
                }
            """)
            return
        
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if not self.capture.isOpened():
                self.status_label.setText("❌ Camera not available")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #ff6b6b;
                        font-size: 10px;
                        padding: 2px;
                    }
                """)
                return
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Start timer for frame updates
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_frame)
            self.timer.start(1000 // self.fps)  # Convert FPS to milliseconds
            
            self.status_label.setText(f"✅ Camera {self.camera_index} active ({self.fps} FPS)")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4ade80;
                    font-size: 10px;
                    padding: 2px;
                }
            """)
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 10px;
                    padding: 2px;
                }
            """)
            print(f"⚠️  Webcam viewport error: {e}")
    
    def _update_frame(self):
        """Capture and display a frame from the webcam"""
        if not self.capture or not self.capture.isOpened():
            return
        
        ret, frame = self.capture.read()
        
        if ret:
            # Convert BGR (OpenCV) to RGB (Qt)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to viewport size while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """Clean up camera resources when closing"""
        self._cleanup()
        super().closeEvent(event)
    
    def _cleanup(self):
        """Release camera and stop timer"""
        if self.timer:
            self.timer.stop()
        
        if self.capture:
            self.capture.release()
            self.capture = None
            
        self.status_label.setText("Camera stopped")
