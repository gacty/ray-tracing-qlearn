import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

GRID_SIZE = 10
OBSTACLES = [(3,3),(3,4),(3,5),(4,5),(5,5),(6,5)]
LIGHTS = [(9,9)]
AGENT_PATH = [(0,0),(1,0),(1,1),(2,1),(3,1),(3,2),(4,2),(5,2),(6,2),
              (6,3),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(8,9),(9,9)]

CELL_SIZE = 1.0
AGENT_RADIUS = 0.3
agent_step = 0

def draw_cube(x,y,color=(1,0,0)):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x+0.5,0.25,y+0.5)
    glutSolidCube(0.5)
    glPopMatrix()

def draw_sphere(x,y,color=(0,0,1)):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x+0.5,AGENT_RADIUS,y+0.5)
    glutSolidSphere(AGENT_RADIUS,20,20)
    glPopMatrix()

def draw_floor():
    glColor3f(0.3,0.3,0.3)
    glBegin(GL_QUADS)
    glVertex3f(0,0.001,0)
    glVertex3f(GRID_SIZE,0.001,0)
    glVertex3f(GRID_SIZE,0.001,GRID_SIZE)
    glVertex3f(0,0.001,GRID_SIZE)
    glEnd()

def display():
    global agent_step
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(GRID_SIZE/2, GRID_SIZE*2, GRID_SIZE*3,
              GRID_SIZE/2, 0, GRID_SIZE/2,
              0,1,0)
    draw_floor()
    for x,y in OBSTACLES:
        draw_cube(x,y,(0.5,0.2,0.2))
    for x,y in LIGHTS:
        draw_cube(x,y,(1,1,0))
    for i in range(agent_step+1):
        x,y = AGENT_PATH[i]
        draw_sphere(x,y,(0,0,1))
    glutSwapBuffers()

def timer(v):
    global agent_step
    agent_step += 1
    if agent_step >= len(AGENT_PATH):
        agent_step = len(AGENT_PATH)-1
    glutPostRedisplay()
    glutTimerFunc(500,timer,1)

def reshape(w,h):
    glViewport(0,0,w,h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45,float(w)/float(h),0.1,100)
    glMatrixMode(GL_MODELVIEW)

def init():
    glClearColor(0.1,0.1,0.1,1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [GRID_SIZE*2, GRID_SIZE*5, GRID_SIZE*2, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1,1,1,1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1,1,1,1])

if __name__=="__main__":
    from OpenGL.GLUT import glutInit
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800,800)
    glutCreateWindow(b"Visible Q-Learning Agent 3D")
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutTimerFunc(500,timer,1)
    glutMainLoop()
