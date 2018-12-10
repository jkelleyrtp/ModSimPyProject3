import numpy as np
import matplotlib.pyplot as plt
import cupy
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import scipy as sp
from emojipy import Emoji
import imageio



###############################################################################
# This is a 2D computational fluid dynamics solver implementing a simple      #
# Navier-Stokes metho̧d. To interface, simpl̷y input two arr̉̈́̌͋́̊ͮays of the o̗̳͉̗̲͚͂ͅb̰̖̝͚stacle #
# in counterclo̢͘ckwise dir̷̴҉e̵c̵̨͠tions. T҉͞h̶͜e solver does̴ its pers̡̕͘onal̸ best t̤̰̤̤͎o͓̙̻̳̫̠͉͙ find  #
# a̺̤̼̻̬ ̘̰̱̜̖̦̬͎s̭̙̠o̥̜l̬̥͚̲͎u͙̹̬̩̘̬t͓͍̙͖̬ͅio͎n̻ to flow around the o͎̰̦̘͖̦ͅb̬̗͇̭̹s͔̼̹̹t̯̭̥̭͓̼̼ͅacle. C̀ò̆ͤd̑͋̈́̉̊ͭ̑ĕͭͩ̾͗ derived frő̈ͫ̚m PyCFD e̞̱̯̻͈̺x̯̤̰͈a͕̳m͓̞̻͖̰p̣̘̬͇̻̳̤̺l̠̘̻e̯̭͖̩̰̤͎̗ͅs͇̙̹͖ and #
# t̰̣̖̩̬͛̒͂̉͋̔͌̒̚h͙̠̹͈̻̱͕̅͗̓̔̂ͩ̚e ̜̻͚̲ͯ̽ͩ̊ͫd̫̊͆̌̀̐̃ḁ̤̤̗͌̔̇̾r̙ͥ͒̔ͮ̆̈́̈̉̚k ̗̜͓̖̠̎̾ͮl̩̠̬͎̫̺̜̥̍͑or̲̮̜͙̭͌ͅd hih͉̞̣̱̥̫ͪ̎ͤ̑ͧȉ̗̦̳͇̼̦̯ͨ̿̋ͩ̑̾m̥̬͉̦͈͔̾̆̌ͦͦͨ̆ͅs̟͙̭͔͕ͧ̓ͬ̈̃͑e̜͖̤̫̣͌ͥ̓̚l͓̺̳̞͚ͨ̄̀͊̌̊̒f̺͕̮̰̤̋ͩ͑ͩͤͨͪ̈̑. B͡҉ew̴a̴r͟͟e a͜҉;l̶̛l m̆̋̃̐o̸̎̀͢rͤ̑ͬ̌ͣ͑͜͟t̵̏ͤ͛̋ͮ̚͠al̶̪̼̠͎̜̘̙̭͋̃̆͂ͫͥͬ͡ s̴̶̻̹̗̮̬̥͉̭̉ͮͬ͗̃̒ͭô̴͓̯͑̍ͧ̂͌̀̚u̥̥͔͖̪̝͚͕̔ͪ̃ͬ̓l̺͍̈́̿͡ wͬ͐̽̉̌̈҉̷̖͉͡ḩ̣͍͂o̵̟͈̲̠͙͎̪͖̎̂ͥͫ̈́͛̆ ̝̣͍̘͈̈́̅͠e̗̦̦̠͖͓̒̈́ͮ̈́ͥ̑͒ͩ͟nͣͮ̆ͨ̃ͩ͐͋ͬ͡͏̩̗̟̟t̠̞̫̤̘̪͓̙̫̽͑ę͎͎̜͉̰̔ͭr̹̞̣̓̆̐ͤ̐̒̀̏̕ ț̝͖̤͇̩̙͚͉͔̭̪͕͇͆ͥͫ̿ͥ̑͒̐̂͋͆͊ͬ̋ͦͮ̈̓̀͘ͅḩ̢̛́ͦ̎̓̐̂ͧ͛̚͏͉̝͔͇̣̞̹̝̱͚̭̟͇̱̳î̴͈̼̞͚̰̜̫͔͉̱̲̤̫͔̠̻͑̔͆̊ͥͯ̏̀ͅs̸̮̭͙͚̝̪̬̭̗͇̼̩̱̹̭̹ͫ́̑̎̆̊ͧͤͦͅͅͅ ̧̨̨͎̗̖͕͚͔̟̳̬̤̰̿̈ͪ̎̔ͯ͌ͬ̚r͓͉̤̝̘̲͓̳̱̖͒͐͛͊̃͒̎̆̏ͬͦͧ̏͒͢͡͞ͅe̶̵̹̼̮͚̹̳̲̦̦̞̪͌͒ͦͯͦ͂̓̿͑̇ͬ͢͜͠a̾̓̋ͦ͋ͣ̈͋ͬ̄͒̋́̾̂̅͒͢͜҉̖̠͔̝̰̙̩̜͚̳̼͈l̷͉̫̠͍̯̈́̍̌̆ͪ̄̽͋͂̐ͬ͆ͥ̌́͟m̴͔̹̜̥̱̦̫̘̮̭̻̖̰̖̦͎͓͎̫͆̔̾̂̿̽̓͊́̒̀͢͡            #
###############################################################################

emojis = [
    '🐔',
    '🐄',
    '🐐',
    '🐖',
    '🐑',
    '🐟'
]

def emoji2cfd(emoji, emoji_width = 1):
    '''
    Converts the inputted emoji to numpy array and feeds it into image2cdf.
    The emoji_width parameter helps scale the emoji properly for the cfd solver

    '''

    emoji_img_item = Emoji.to_image(emoji)
    start = a.find("src=")
    link = a[start+5: -3]
    im = imageio.imread(link)


def emoji2image(emoji):
    a = Emoji.to_image(emoji)
    url = a[a.find("src=")+5:a.find("/>")-1]
    image = imageio.imread(url)
    return image

import cv2
def img2contour(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img = cv2.inRange(img, (1, 0, 0), (180, 255, 255))
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(img, mode=mode, method=method)
    #print(contours[0])
    #while True:
    #    cv2.imshow('e', img)
    #    if cv2.waitKey(0) & 0xFF == ord('q'):
    #        break
    #cv2.destroyAllWindows()
    return contours


def gen_shapes():
    x = np.array([1.00000, 0.95041, 0.90067, 0.80097, 0.70102, 0.60085,
                        0.50049, 0.40000, 0.29875, 0.24814, 0.19761, 0.14722,
                        0.09710, 0.07217, 0.04742, 0.02297, 0.01098, 0.00000,
                        0.01402, 0.02703, 0.05258, 0.07783, 0.10290, 0.15278,
                        0.20239, 0.25186, 0.30125, 0.40000, 0.49951, 0.59915,
                        0.69898, 0.79903, 0.89933, 0.94959, 1.00000, 1.00000])
    y = np.array([0.00105, 0.00990, 0.01816, 0.03296, 0.04551, 0.05580,
                        0.06356, 0.06837, 0.06875, 0.06668, 0.06276, 0.05665,
                        0.04766, 0.04169, 0.03420, 0.02411, 0.01694, 0.00000,
                        -0.01448, -0.01927, -0.02482, -0.02809, -0.03016, -0.03227,
                        -0.03276, -0.03230, -0.03125, -0.02837, -0.02468, -0.02024,
                        -0.01551, -0.01074, -0.00594, -0.00352, -0.00105, 0.00105])

    x_1 = np.linspace(0, 1, 50)
    x_2 = np.linspace(1, 1, 50)
    x_3 = np.linspace(1, 0, 50)
    x_4 = np.linspace(0, 0, 50)

    y_1 = np.linspace(0, 0, 50)
    y_2 = np.linspace(0, 1, 50)
    y_3 = np.linspace(1, 1, 50)
    y_4 = np.linspace(1, 0, 50)

    x_list2 = np.concatenate([x_1, x_2, x_3, x_4])
    y_list2 = np.concatenate([y_1, y_2, y_3, y_4])

    x, y = x_list2, y_list2
    #return np.array([0]), np.array([0]

    return x, y

def getPressure(xData, yData, scale, div=20, aoa=0, xShift=0, yShift=0, plot=False):
    xData = xData.copy()
    yData = yData.copy()

    # ===========================================================================
    # Calculation of geometric properties of boundary element segments
    # ===========================================================================
    def geometry(x_list, y_list, seg_list):
        Ns = int(np.sum(seg_list))  # total no. of segments
        Np = Ns + 1  # total no. of segment end-points

        lb = np.sqrt((x_list[1:] - x_list[:-1]) ** 2 + (y_list[1:] - y_list[:-1]) ** 2)

        # total no. of segments at the beginning of each boundary element
        seg_num = np.zeros(seg_list.size)
        for i in range(1, seg_list.size):
            seg_num[i] = seg_num[i - 1] + seg_list[i - 1]

        x = np.zeros(Np)
        y = np.zeros(Np)
        x[0] = x[-1] = x_list[0];
        y[0] = y[-1] = y_list[0]
        for i in range(seg_list.size):
            x[int(seg_num[i]):int(seg_num[i] + seg_list[i] + 1)] = np.linspace(x_list[i], x_list[i + 1], seg_list[i] + 1)
            y[int(seg_num[i]):int(seg_num[i] + seg_list[i] + 1)] = np.linspace(y_list[i], y_list[i + 1], seg_list[i] + 1)

        # mid-pt of segments
        xm = 0.5 * (x[1:] + x[:-1])
        ym = 0.5 * (y[1:] + y[:-1])

        # list of mid-pts by boundary element index
        xms, yms = [[0] * seg_list.size for i in range(2)]  # sequence with 1 element for each segment
        for i in range(seg_list.size):
            xms[i] = np.array(xm[int(seg_num[i]):int(seg_num[i] + seg_list[i])])
            yms[i] = np.array(ym[int(seg_num[i]):int(seg_num[i] + seg_list[i])])

        # length of segments
        l = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)

        # normal vectors
        ny = (x[:-1] - x[1:]) / l
        nx = (y[1:] - y[:-1]) / l
        return x, y, xm, ym, xms, yms, nx, ny, l, Ns, seg_num, lb


    # ===========================================================================
    # Setting boundary conditions for each segement of boundary element
    # ===========================================================================
    def setBC(bct, bcv, seg_list, seg_num):
        BCT, BCV = [np.zeros(Ns) for i in range(2)]
        for i in range(seg_list.size):
            BCT[int(seg_num[i]):int(seg_num[i] + seg_list[i])] = bct[i]
            BCV[int(seg_num[i]):int(seg_num[i] + seg_list[i])] = bcv[i]

        return BCT, BCV


    # ===========================================================================
    # Calculate integral coefficients F1 & F2 for a given array of points (x0,y0)
    # ===========================================================================
    from time import time


    def F1F2(x0, y0, x, y, l, nx, ny):
        t0 = time()
        k = int(Ns)  # no. of segments
        s = x0.size  # no. of points

        A, B, E, F1, F2 = [np.zeros((k, s)) for i in range(5)]
        k = np.arange(k)
        s = np.arange(s)
        K, S = np.meshgrid(k, s)

        t0 = time()
        f = np.square(l[K]).T
        A[K, :] = f
        # print(l)
        # print(K)
        # print(l[K])
        # print(round(time()-t0, 4)); t0=time()
        #print(l.shape)
        #print(K.shape)
        #print(A[K, :].shape)
        B[K, S] = 2 * l[K] * (-(x[K] - x0[S]) * ny[K] + (y[K] - y0[S]) * nx[K])
        E[K, S] = (x[K] - x0[S]) ** 2 + (y[K] - y0[S]) ** 2

        M = 4 * A * E - B ** 2
        D = 0.5 * B / A

        zero = 1e-10  # a very small number to take care of floating point errors
        # Jth point (x0[J],y0[J]) intersects the (extended) Ith line segment
        I, J = np.where(M < zero)
        # jth point (x0[j],y0[j]) does not intersect (extended) ith line segment
        i, j = np.where(M > zero)

        # for M = 0 (lim D->0 D*ln(D)=0 )
        # since the log function cannot handle log(0), 'zeros' have been added to log(D) -> log(D+zero)
        F1[I, J] = 0.5 * l[I] * (np.log(l[I]) \
                                 + (1 + D[I, J]) * np.log(np.abs(1 + D[I, J]) + zero) \
                                 - D[I, J] * np.log(np.abs(D[I, J] + zero)) - 1) / np.pi
        # for M > 0
        H = np.arctan((2 * A[i, j] + B[i, j]) / np.sqrt(M[i, j])) - np.arctan(B[i, j] / np.sqrt(M[i, j]))
        F1[i, j] = 0.25 * l[i] * (2 * (np.log(l[i]) - 1) \
                                  - D[i, j] * np.log(np.abs(E[i, j] / A[i, j])) \
                                  + (1 + D[i, j]) * np.log(np.abs(1 + 2 * D[i, j] + E[i, j] / A[i, j])) \
                                  + H * np.sqrt(M[i, j]) / A[i, j]) / np.pi
        F2[i, j] = l[i] * (nx[i] * (x[i] - x0[j]) + ny[i] * (y[i] - y0[j])) * H / np.sqrt(M[i, j]) / np.pi

        return F1.T, F2.T


    # ===========================================================================
    # Build matrix system from F1 & F2 to find remaining BCs
    # ===========================================================================
    def pqBC(F1, F2, BCT, BCV):
        Ns = BCT.size
        F2x = F2 - 0.5 * np.eye(Ns)
        a, b = [np.zeros((Ns, Ns)) for i in range(2)]

        # phi is known - d(phi)/dn is unknown
        col_p = np.where(BCT == 0)
        a[:, col_p] = -F1[:, col_p]
        b[:, col_p] = -F2x[:, col_p]
        # d(phi)/dn is known - phi is unknown
        col_q = np.where(BCT == 1)
        a[:, col_q] = F2x[:, col_q]
        b[:, col_q] = F1[:, col_q]

        BCV2 = np.linalg.solve(a, np.dot(b, BCV))

        p = BCV2.copy()
        q = BCV2.copy()

        p[col_p] = BCV[col_p]  # replace with known 'phi's
        q[col_q] = BCV[col_q]  # replace with known 'd(phi)/dn's

        return p, q



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # GEOMETRY
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # (x,y)       segment end-points
    # (xm, ym)    segment mid-points
    # (xms,yms)   segment mid-point grouped by boundary elements
    # (nx,ny)     normal vector components centered at (xm,ym)
    # l           segment lengths
    # Ns          total no. of segments
    # seg_num     total no. of segments at the end of each boundary element
    # lb          length of boundary element

    # End-point coordinates of external rectangular domain (anti-clockwise / last pt = first pt)
    # x_list1 = np.array([-10.,50.,50.,-10.,-10.])
    # y_list1 = np.array([-20.,-20.,20,20,-20])
    x_list1 = np.array([-10., 50., 50., -10., -10.])/2
    y_list1 = np.array([-20., -20., 20, 20, -20])/2

    x_list1 -= np.average(x_list1)
    y_list1 -= np.average(y_list1)
    # No. of segments for each boundary element
    seg_list1 = np.array([40, 20, 40, 20])
    # Indices
    #inlet = 3  # inlet
    inlet = 3  # inlet
    outlet = 1  # outlet

    # Coordinates of airfoil (clockwise / last pt = first pt)
    #scale = 15
    transX = xShift
    transY = yShift
    #x_list2 = scale * np.array([1.00000, 0.95041, 0.90067, 0.80097, 0.70102, 0.60085,
    #                            0.50049, 0.40000, 0.29875, 0.24814, 0.19761, 0.14722,
    #                            0.09710, 0.07217, 0.04742, 0.02297, 0.01098, 0.00000,
    #                            0.01402, 0.02703, 0.05258, 0.07783, 0.10290, 0.15278,
    #                            0.20239, 0.25186, 0.30125, 0.40000, 0.49951, 0.59915,
    #                            0.69898, 0.79903, 0.89933, 0.94959, 1.00000, 1.00000])[::-1]  # clockwise
    #y_list2 = scale * np.array([0.00105, 0.00990, 0.01816, 0.03296, 0.04551, 0.05580,
    #                            0.06356, 0.06837, 0.06875, 0.06668, 0.06276, 0.05665,
    #                            0.04766, 0.04169, 0.03420, 0.02411, 0.01694, 0.00000,
    #                            -0.01448, -0.01927, -0.02482, -0.02809, -0.03016, -0.03227,
    #                            -0.03276, -0.03230, -0.03125, -0.02837, -0.02468, -0.02024,
    #                            -0.01551, -0.01074, -0.00594, -0.00352, -0.00105, 0.00105])[::-1]  # clockwise

    #x_list2, y_list2 = gen_shapes()
    x_list2, y_list2 = xData, yData
    x_list2 = x_list2[::-1]
    y_list2 = y_list2[::-1]
    x_list2 *= scale
    y_list2 *= scale

    x_list2 -= np.average(x_list2)
    y_list2 -= np.average(y_list2)
    x_list2 += transX
    y_list2 += transY

    alpha = -aoa # angle of attack
    # alpha = 45 # angle of attack
    D2R = np.pi / 180
    ang = np.radians(alpha)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    img = np.c_[x_list2, y_list2]
    #print(img)
    rotated = np.matmul(rot,img.T)
    x_list2 = rotated[0]
    y_list2 = rotated[1]
    #x_list2 -= np.average(x_list2)
    #y_list2 -= np.average(y_list2)

    #x_list2 -= (x_list2.max()-x_list2.min())/2
    #y_list2 -= (y_list2.max()-y_list2.min())/2

    #x_list2 = np.dot(np.c_[x_list2, y_list2], rot)[:, 0]
    #y_list2 = np.dot(np.c_[x_list2, y_list2], rot)[:, 1]
    Ns2 = x_list2.size - 1
    seg_list2 = np.ones(Ns2)

    projY = y_list2
    frontal_area = projY.max() - projY.min()
    #print(frontal_area)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    x1, y1, xm1, ym1, xms1, yms1, nx1, ny1, l1, Ns1, seg_num1, lb1 = geometry(x_list1, y_list1, seg_list1)
    x2, y2, xm2, ym2, xms2, yms2, nx2, ny2, l2, Ns2, seg_num2, lb2 = geometry(x_list2, y_list2, seg_list2)

    # Combining the internal & external boundaries
    x = np.append(x1[:-1], x2[:-1])
    y = np.append(y1[:-1], y2[:-1])
    xm = np.append(xm1, xm2)
    ym = np.append(ym1, ym2)
    l = np.append(l1, l2)
    nx = np.append(nx1, nx2)
    ny = np.append(ny1, ny2)
    Ns = Ns1 + Ns2
    seg_list = np.append(seg_list1, seg_list2)
    seg_num = np.zeros(seg_list.size)
    for i in range(1, seg_list.size):
        seg_num[i] = seg_num[i - 1] + seg_list[i - 1]

    if(plot):
    	fig = plt.figure(figsize=(12, 12), dpi=100)
    	fig.add_subplot(111, aspect='equal')
    	plt.scatter(x, y, c=u'r', marker=u'o')
    	# plt.scatter(xm,ym,c=u'g',marker=u'^')
    	# plt.quiver(xm,ym,nx,ny)
    	plt.title('Boundary, Segments & Normal Vectors')
    	plt.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # BOUNDARY CONDITIONS
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    U = 5.  # volume flow rate
    bct = np.ones(Ns)  # sequence of boundary condition types: 0->p, 1->q
    bcv = np.zeros(Ns)  # sequence of boundary condition values
    bcv[inlet] = -U / lb1[inlet]
    bcv[outlet] = U / lb1[outlet]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    BCT, BCV = setBC(bct, bcv, seg_list, seg_num)

    F1, F2 = F1F2(xm, ym, x, y, l, nx, ny)  # obtaining F1 & F2 for segment mid-points
    p, q = pqBC(F1, F2, BCT, BCV)  # solving for additional boundary conditions

    # Generating internal points (excludes boundary)
    xDim = 1
    yDim = 1
    mul = div
    Nx = xDim * mul;
    Ny = yDim * mul;
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    X = np.linspace(x.min(), x.max(), Nx)
    Y = np.linspace(y.min(), y.max(), Ny)
    X, Y = np.meshgrid(X[1:-1], Y[1:-1])

    X = X.ravel();
    Y = Y.ravel()

    # Determines points within airfoil region by computing dot product [D] of
    # normal vectors and vectors from mid-points on airfoil surface to points
    # of interest. If all the dot products are positive, point of interest
    # lies within the airfoil. Also, removes points on the outside which are a
    # with a certain minimum distance [L] from the airfoil.
    D = np.zeros((X.size, xm2.size))
    L = np.zeros((X.size, xm2.size))
    I = []

    remove_dist = 1  # min dist from point to obstacle
    for i in range(X.size):
        for j in range(xm2.size):
            D[i, j] = (X[i] - xm2[j]) * nx2[j] + (Y[i] - ym2[j]) * ny2[j]
            L[i, j] = np.sqrt((X[i] - xm2[j]) ** 2 + (Y[i] - ym2[j]) ** 2)
        if ((D[i, :] > 0).all()):
            I.append(i)
        elif ((L[i, :] < remove_dist).any()):
            I.append(i)

    X_2 = X
    Y_2 = Y
    X = np.delete(X.ravel(), I)
    Y = np.delete(Y.ravel(), I)

    if plot:
        fig = plt.figure(figsize=(12, 12), dpi=100)
        fig.add_subplot(111, aspect='equal')
        plt.fill(x1, y1, fill=False, lw=3)
        plt.fill(x2, y2, fill=True, lw=3)
        plt.scatter(X, Y, c=u'b', marker=u'*')
        plt.title(r'Internal Points for Calculation of $\phi$, $u$ & $v$')
        plt.show()

    # Calculate velocity (u,v) at internal grid points (X,Y)
    # ===========================================================================
    delta_x = delta_y = 0.05
    F1, F2 = F1F2(X + delta_x, Y, x, y, l, nx, ny)
    phi_x_plus = (np.dot(F2, p) - np.dot(F1, q))
    F1, F2 = F1F2(X - delta_x, Y, x, y, l, nx, ny)
    phi_x_minus = (np.dot(F2, p) - np.dot(F1, q))
    F1, F2 = F1F2(X, Y + delta_y, x, y, l, nx, ny)
    phi_y_plus = (np.dot(F2, p) - np.dot(F1, q))
    F1, F2 = F1F2(X, Y - delta_y, x, y, l, nx, ny)
    phi_y_minus = (np.dot(F2, p) - np.dot(F1, q))

    # Central difference to determine velocity
    u = 0.5 * (phi_x_plus - phi_x_minus) / delta_x
    v = 0.5 * (phi_y_plus - phi_y_minus) / delta_y

    P = -0.5*(u*u + v*v) # Bernoulli static pressure equation

    from scipy.interpolate import interp1d, griddata
    nx, ny = 100, 100
    pts = np.vstack((X, Y)).T
    vals = np.vstack((u, v)).T

    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)

    ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T

    ivals = griddata(pts, vals, ipts, method='cubic')
    ui, vi = ivals.T
    ui.shape = vi.shape = (ny, nx)

    if plot:
        fig = plt.figure(figsize=(12, 12), dpi=100)
        fig.add_subplot(111, aspect='equal')
        plt.fill(x1, y1, fill=False, lw=3)
        plt.fill(x2, y2, fill=True, lw=3)
        plt.quiver(X[::3], Y[::3], u[::3], v[::3])
        plt.title('CFD')
        plt.show()
    speed = np.sqrt(ui*ui + vi*vi)
    #plt.streamplot(xi, yi, ui, vi, density=2.0)
    #plt.contourf(xi, yi, vi)
    pairs = []
    for i in range(len(X)):
        pairs.append([X[i], Y[i], P[i]])

    pairs = np.asarray(pairs)

    pFrontSum = 0
    pTopSum = 0
    pUnderSum = 0

    for i in pairs:
        X_l, Y_l, P_l = i
        if X_l < (x_list2.max() - x_list2.min())/2:
            # Point is in front of the object
            if y_list2.min() <= Y_l < y_list2.max():
                # Point is directly in front of the object
                pFrontSum += P_l

        if Y_l > (y_list2.max() - y_list2.min())/2:
            # Point is above object
            if x_list2.min() <= X_l <= x_list2.max():
                # Point is directly above object
                pTopSum += P_l
        else:
            # Point is below object
            if x_list2.min() <= X_l <= x_list2.max():
                # Point is directly above object
                pUnderSum += P_l

    #plt.contourf(X.reshape(Nx,Ny),Y.reshape(Nx,Ny),P,15,alpha=0.5)


    U_ideal = 0.2498653558682373
    V_ideal = 2.635814773584536e-12
    E_ideal = 0.2498660226708361

    dragArea = y_list2.max() - y_list2.min()
    liftArea = x_list2.max() - x_list2.min()

    U_actual = np.average(u)
    V_actual = np.average(v)
    E_actual = np.average(np.sqrt(u**2 + v**2))

    UDropPerArea = (U_ideal-U_actual)/dragArea
    VDropPerArea = (V_ideal-V_actual)/dragArea
    EDropPerArea = (E_ideal-E_actual)/dragArea

    #print("Average u" , U_actual)
    #print("Average v" , V_actual)
    #print("Average energy" , E_ideal)
    #
    #print("Ideal u" , U_ideal)
    #print("Ideal v" , V_ideal)
    #print("Ideal energy" , E_ideal)
    #
    #print("U drop per area", UDropPerArea)
    #print("V drop per area", VDropPerArea)
    #print("E drop per area", EDropPerArea)
    #
    #print("Front pressure", pFrontSum)
    #print("Top pressure", pTopSum)
    #print("Bottom pressure", pUnderSum)
    #print("Front pressure per area", pFrontSum/dragArea)
    #print("Top pressure per area", pTopSum/liftArea)
    #print("Bottom pressure per area", pUnderSum/liftArea)
    #print("Top/bottom differential", (pTopSum/liftArea)-(pUnderSum/liftArea))


    # Drag Force = Cd * rho * V^2 * A
    #              ------------------
    #                      2

    return pFrontSum/dragArea, (pTopSum/liftArea)-(pUnderSum/liftArea)

    # Avg U: 0.2498653558682373
    # Avg V: 2.635814773584536e-12
    # Avg energy: 0.2498660226708361


    #X, Y = np.meshgrid(X, Y)
    #plt.contourf(X, Y, u, 15, alpha=0.5)
    #plt.show()
    #plt.contourf(X, Y, v, 15, alpha=0.5)
    #plt.show()


from multiprocessing import Pool

def eval_emoji(emoji, r=[180], plot=True):
        image = emoji2image(emoji)
        contours = img2contour(image)
        contours = contours[0]
        contours = contours.reshape((contours.shape[0], contours.shape[-1]))
        contours = contours / contours.max()
        M = cv2.moments(contours)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        contours = [list(i) for i in contours]
        contours.append(contours[0])
        xList = [i[0] for i in contours]
        yList = [i[1] for i in contours]
        xList = np.asarray(xList)[::-1]
        yList = np.asarray(yList)[::-1]
        xList -= cx
        yList -= cy
        tmp = []
        for a in r:
            print(a)
            drag, lift = getPressure(xList, yList, scale=7.5, aoa=a, div=30, xShift=0, yShift=0, plot=True)
            tmp.append([a, drag, lift])
        return [emoji, tmp]

if __name__ == '__main__':

    eval_emoji('🐖')

    #p = Pool(8)

    #data = p.map(eval_pool, emojis)

    #print(data)
    #for i in data:
    #    print(i[0])
    #    for j in i[1]:
    #        print("\t", j[0], ":")
    #        print("\t\tcd:", j[1])
    #        print("\t\tcl:", j[2])
