import RPi.GPIO as GPIO
import time

Motor_R1_Pin = 16
Motor_R2_Pin = 18
Motor_L1_Pin = 11
Motor_L2_Pin = 13
t = 0.1
dc = 30

# 蜂鳴器
BUZZ_PIN = 15
reference = "https://atceiling.blogspot.com/2014/03/raspberry-pi_18.html#.U2zM_4GSzeB"  # 超聲波參考資料
# 超聲波
trigger_pin = None
echo_pin = None
# --GPIO設定----------------------------------------------
GPIO.setmode(GPIO.BOARD)
GPIO.setup(Motor_R1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_R2_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L2_Pin, GPIO.OUT, initial=GPIO.LOW)

GPIO.setup(BUZZ_PIN, GPIO.OUT)

GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)
# ---------------------------------------------------------

pwm = GPIO.PWM(BUZZ_PIN, 262)  # 設定蜂鳴器腳位+音高
pwm.start(0)

pwm_r1 = GPIO.PWM(Motor_R1_Pin, 500)
pwm_r2 = GPIO.PWM(Motor_R2_Pin, 500)
pwm_l1 = GPIO.PWM(Motor_L1_Pin, 500)
pwm_l2 = GPIO.PWM(Motor_L2_Pin, 500)
pwm_r1.start(0)
pwm_r2.start(0)
pwm_l1.start(0)
pwm_l2.start(0)


def stop():
    pwm_r1.ChangeDutyCycle(0)
    pwm_r2.ChangeDutyCycle(0)
    pwm_l1.ChangeDutyCycle(0)
    pwm_l2.ChangeDutyCycle(0)


def forward():
    pwm_r1.ChangeDutyCycle(dc)
    pwm_r2.ChangeDutyCycle(0)
    pwm_l1.ChangeDutyCycle(dc)
    pwm_l2.ChangeDutyCycle(0)
    time.sleep(t)
    stop()


def backward():
    pwm_r1.ChangeDutyCycle(0)
    pwm_r2.ChangeDutyCycle(dc)
    pwm_l1.ChangeDutyCycle(0)
    pwm_l2.ChangeDutyCycle(dc)
    time.sleep(t)
    stop()


def turnLeft():
    pwm_r1.ChangeDutyCycle(dc)
    pwm_r2.ChangeDutyCycle(0)
    pwm_l1.ChangeDutyCycle(0)
    pwm_l2.ChangeDutyCycle(0)
    time.sleep(t)
    stop()


def turnRight():
    pwm_r1.ChangeDutyCycle(0)
    pwm_r2.ChangeDutyCycle(0)
    pwm_l1.ChangeDutyCycle(dc)
    pwm_l2.ChangeDutyCycle(0)
    time.sleep(t)
    stop()


def cleanup():
    stop()
    pwm_r1.stop()
    pwm_r2.stop()
    pwm_l1.stop()
    pwm_l2.stop()
    GPIO.cleanup()


def play_buzz():
    pwm.ChangeDutyCycle(50)
    time.sleep(0.3)
    pwm.ChangeDutyCycle(0)


#  超聲波--------------------------------------------------
def send_trigger_pulse():
    GPIO.output(trigger_pin, True)
    time.sleep(0.001)
    GPIO.output(trigger_pin, False)


def wait_for_echo(value, timeout):
    count = timeout
    while GPIO.input(echo_pin) != value and count > 0:
        count = count - 1


def get_distance():
    send_trigger_pulse()
    wait_for_echo(True, 5000)
    start = time.time()
    wait_for_echo(False, 5000)
    finish = time.time()
    pulse_len = finish - start
    distance_cm = pulse_len * 340 *100 /2
    distance_in = distance_cm / 2.5
    return distance_cm, distance_in


def print_distance():
    print("cm=%f\tinches=%f" % get_distance())
    time.sleep(0.5)



