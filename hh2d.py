#!/usr/bin/python3
# -*- coding: utf-8 -*-
# last tested Python version: 3.6.9
# Hodgkin-Huxley model on a 2D lattice
# FvW 03/2018

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def hh2d(N, T, t0, dt, s, D, C, gNa, gK, gL, VNa, VK, VL, I0, stim, blocks):
    #C1 = 1.0/C
    # initialize Hodgkin-Huxley system
    v = -10*np.ones((N,N))
    m = np.zeros((N,N))
    n = np.zeros((N,N))
    h = np.zeros((N,N))
    dvdt = np.zeros((N,N))
    dmdt = np.zeros((N,N))
    dndt = np.zeros((N,N))
    dhdt = np.zeros((N,N))
    s_sqrt_dt = s*np.sqrt(dt)
    X = np.zeros((T,N,N))
    X[0,:,:] = v
    #offset = 0 # Int(round(1*nt))
    # stimulation protocol
    I = np.zeros((t0+T,N,N))
    #I = I0*np.ones((t0+T,N,N))
    for st in stim:
        t_on, t_off = st[0]
        x0, x1 = st[1]
        y0, y1 = st[2]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] = I0

    # iterate
    for t in range(1, t0+T):
        if (t%100 == 0): print("    t = ", t, "/", t0+T, "\r", end="")
        # HH equations
        # dV/dt (membrane potential)
        dvdt = 1/C*(-gNa*(m**3)*h*(v-VNa) \
                    -gK*(n**4)*(v-VK) \
                    -gL*(v-VL) + I[t,:,:]) + D*L(v)
        # dm/dt (Na+ channel activation)
        alpha_m = 0.1*(-v+25)/(np.exp((-v+25)/10)-1)
        beta_m = 4 * np.exp(-v/18)
        dmdt = (alpha_m * (1 - m) - beta_m * m)
        # dh/dt (Na+ channel inactivation)
        alpha_h = 0.07 * np.exp(-v/20)
        beta_h = 1 / (np.exp((-v + 30) / 10) + 1)
        dhdt = (alpha_h * (1 - h) - beta_h*h)
        # dn/dt (K+ channel gating)
        alpha_n = 0.01 * (-v + 10)/(np.exp((-v + 10)/10) - 1)
        beta_n = 0.125 * np.exp(-v/80)
        dndt = (alpha_n * (1 - n) - beta_n*n)
        # Ito stochastic integration
        v += (dvdt*dt + s_sqrt_dt*np.random.randn(N,N))
        m += (dmdt*dt)
        n += (dndt*dt)
        h += (dhdt*dt)
        # dead block(s):
        for bl in blocks:
            v[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
            m[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
            n[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
            h[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
        # store
        if (t >= t0):
            X[t-t0,:,:] = v
    print("\n")
    return X


def animate_video(fname, x):
    # BW
    y = 255 * (x-x.min()) / (x.max()-x.min())
    # BW inverted
    #y = 255 * ( 1 - (x-x.min()) / (x.max()-x.min()) )
    y = y.astype(np.uint8)
    nt, nx, ny = x.shape
    print("nt = {nt:d}, nx = {nx:d}, ny = {ny:d}")
    # write video using opencv
    frate = 30
    out = cv2.VideoWriter(fname, \
                          cv2.VideoWriter_fourcc(*'mp4v'), \
                          frate, (nx,ny))
    for i in range(0,nt):
        print(f"i = {i:d}/{nt:d}\r", end="")
        img = np.ones((nx, ny, 3), dtype=np.uint8)
        for j in range(3): img[:,:,j] = y[i,::-1,:]
        out.write(img)
    out.release()
    print("")


def L(x):
    # Laplace operator
    # periodic boundary conditions
    xU = np.roll(x, shift=-1, axis=0)
    xD = np.roll(x, shift=1, axis=0)
    xL = np.roll(x, shift=-1, axis=1)
    xR = np.roll(x, shift=1, axis=1)
    Lx = xU + xD + xL + xR - 4*x
    # non-periodic boundary conditions
    Lx[0,:] = 0.0
    Lx[-1,:] = 0.0
    Lx[:,0] = 0.0
    Lx[:,-1] = 0.0
    return Lx


def main():
    print("Hodgkin-Huxley (HH) lattice model\n")
    N = 128
    T = 10000
    t0 = 2000
    dt = 0.01
    s = 1.0
    D = 1
    # H-H
    C = 1
    gNa = 120
    gK = 36
    gL = 0.3
    VNa = 115
    VK = -12
    VL = 10.6
    I = 45.0
    print("[+] Lattice size N: ", N)
    print("[+] Time steps T: ", T)
    print("[+] Warm-up steps t0: ", t0)
    print("[+] Integration time step dt: ", dt)
    print("[+] Noise intensity s: ", s)
    print("[+] Diffusion coefficient D: ", D)
    print("[+] HH parameter gNa: ", gNa)
    print("[+] HH parameter gK: ", gK)
    print("[+] HH parameter gL: ", gL)
    print("[+] HH parameter VNa: ", VNa)
    print("[+] HH parameter VK: ", VK)
    print("[+] HH parameter VL: ", VL)
    print("[+] HH parameter C: ", C)
    print("[+] Stimulation current I: ", I)

    # stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
    stim = [ [[50,350], [1,5], [1,5]],
             [[1900,2200], [25,30], [1,15]] ]
    #stim = []

    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    blocks = [ [[1,15], [5,10]] ]
    #blocks = []

    # run simulation
    data = hh2d(N, T, t0, dt, s, D, C, gNa, gK, gL, VNa, VK, VL, I, stim, blocks)
    print("[+] Data dimensions: ", data.shape)

    # plot mean voltage
    m = np.mean(np.reshape(data, (T,N*N)), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(m, "-k")
    plt.tight_layout()
    plt.show()

    # save data
    #fname1 = f"hh2d_I_{I:.2f}_s_{s:.2f}_D_{D:.2f}.npy"
    #np.save(fname1, data)
    #print("[+] Data saved as: ", fname1)

    # video
    fname2 = f"hh2d_I_{I:.2f}_s_{s:.2f}_D_{D:.2f}.mp4"
    animate_video(fname2, data)
    print("[+] Data saved as: ", fname2)


if __name__ == "__main__":
    os.system("clear")
    main()
