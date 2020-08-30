import pyautogui
import random
import scipy
import time
from scipy import interpolate


cp = 10  # Number of control points. Must be at least 2.
x1, y1 = pyautogui.position()  # Starting position
x2, y2 = 444, 631  # Destination

# Distribute control points between start and destination evenly.
x = scipy.linspace(x1, x2, num=cp, dtype='int')
y = scipy.linspace(y1, y2, num=cp, dtype='int')

# Randomise inner points a bit (+-RND at most).
RND = 10
xr = scipy.random.randint(-RND, RND, size=cp)
yr = scipy.random.randint(-RND, RND, size=cp)
xr[0] = yr[0] = xr[-1] = yr[-1] = 0
x += xr
y += yr

# Approximate using Bezier spline.
degree = 3 if cp > 3 else cp - 1  # Degree of b-spline. 3 is recommended.
                                  # Must be less than number of control points.
tck, u = scipy.interpolate.splprep([x, y], k=degree)
u = scipy.linspace(0, 1, num=max(pyautogui.size()))
points = scipy.interpolate.splev(u, tck)

# Move mouse.
duration = 0.2
timeout = duration / len(points[0])
for point in zip(*(i.astype(int) for i in points)):
    pyautogui.platformModule._moveTo(*point)
    time.sleep(timeout)
