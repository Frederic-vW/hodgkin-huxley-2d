#!/usr/local/bin/julia
# last tested Julia version: 1.6.1
# Hodgkin-Huxley model on a 2D lattice
# FvW 03/2018

using NPZ
using PyCall
using PyPlot
using Statistics
using VideoIO
@pyimport matplotlib.animation as anim

function hh2d(N, T, t0, dt, s, D, C, gNa, gK, gL, VNa, VK, VL, I0, stim, blocks)
    # initialize Hodgkin-Huxley system
	C1 = 1.0/C
    v = -10*ones(Float64,N,N)
    m = zeros(Float64,N,N)
    n = zeros(Float64,N,N)
    h = zeros(Float64,N,N)
    dvdt = zeros(Float64,N,N)
    dmdt = zeros(Float64,N,N)
    dndt = zeros(Float64,N,N)
    dhdt = zeros(Float64,N,N)
    s_sqrt_dt = s*sqrt(dt)
    X = zeros(Float64,T,N,N)
    # stimulation protocol
    I = zeros(Float64,t0+T,N,N)
    for st in stim
        t_on, t_off = st[1]
        x0, x1 = st[2]
        y0, y1 = st[3]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] .= I0
    end
    # iterate
    for t in range(2, stop=t0+T, step=1)
        (t%100 == 0) && print("    t = ", t, "/", t0+T, "\r")
        # HH equations
        # dV/dt (membrane potential)
        dvdt = C1.*(-gNa.*(m.^3).*h.*(v.-VNa)
                    .- gK.*(n.^4).*(v.-VK)
                    .- gL*(v.-VL) .+ I[t,:,:]) .+ D.*L(v)
        # dm/dt (Na+ channel activation)
        alpha_m = 0.1.*(-v.+25)./(exp.((-v.+25)./10).-1)
        beta_m = 4 * exp.(-v./18)
        dmdt = (alpha_m .* (1 .- m) .- beta_m .* m)
        # dh/dt (Na+ channel inactivation)
        alpha_h = 0.07 * exp.(-v./20)
        beta_h = 1.0 ./ (exp.((-v .+ 30) ./ 10) .+ 1)
        dhdt = (alpha_h .* (1.0 .- h) .- beta_h.*h)
        # dn/dt (K+ channel gating)
        alpha_n = 0.01 .* (-v .+ 10)./(exp.((-v .+ 10)./10) .- 1)
        beta_n = 0.125 .* exp.(-v./80)
        dndt = (alpha_n .* (1.0 .- n) .- beta_n.*n)
        # stochastic integration
        v += (dvdt*dt + s_sqrt_dt*randn(N,N))
        m += (dmdt*dt)
        n += (dndt*dt)
        h += (dhdt*dt)
        # dead block(s):
        for bl in blocks
            v[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
            m[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
            n[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
            h[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
        end
        # store
        (t > t0) && (X[t-t0,:,:] = v)
    end
    println("\n")
    return X
end

function animate_pyplot(fname, data)
    """
    Animate 3D array as .mp4 using PyPlot, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    vmin = minimum(data)
    vmax = maximum(data)
    # setup animation image
    fig = figure(figsize=(6,6))
    axis("off")
    t = imshow(data[1,:,:], origin="lower", cmap=ColorMap("gray"),
			   vmin=vmin, vmax=vmax)
    tight_layout()
    # frame generator
    println("[+] animate")
    function animate(i)
        (i%100 == 0) && print("    t = ", i, "/", nt, "\r")
        t.set_data(data[i+1,:,:])
    end
    # create animation
    ani = anim.FuncAnimation(fig, animate, frames=nt, interval=10)
    println("\n")
    # save animation
    ani[:save](fname, bitrate=-1,
               extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    show()
end

function animate_video(fname, data)
    """
    Animate 3D array as .mp4 using VideoIO, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    # BW
    y = UInt8.(round.(255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    # BW inverted
    #y = UInt8.(round.(255 .- 255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    encoder_options = (color_range=2, crf=0, preset="medium")
    framerate=30
    T = size(data,1)
    open_video_out(fname, y[1,end:-1:1,:], framerate=framerate,
                   encoder_options=encoder_options) do writer
        for i in range(2,stop=T,step=1)
            write(writer, y[i,end:-1:1,:])
        end
    end
end

function L(x)
    # Laplace operator
    # periodic boundary conditions
    xU = circshift(x, [-1 0])
    xD = circshift(x, [1 0])
    xL = circshift(x, [0 -1])
    xR = circshift(x, [0 1])
    Lx = xU + xD + xL + xR - 4x
    # non-periodic boundary conditions
    Lx[1,:] .= 0.0
    Lx[end,:] .= 0.0
    Lx[:,1] .= 0.0
    Lx[:,end] .= 0.0
    return Lx
end

function main()
    println("Hodgkin-Huxley (HH) lattice model\n")
    N = 128
    T = 5000
    t0 = 2000
    dt = 0.01
    s = 1.0
    D = 1.0
    # Hodgkin-Huxley parameters
    C = 1
    gNa = 120
    gK = 36
    gL = 0.3
    VNa = 115
    VK = -12
    VL = 10.6
    I = 45.0
    println("[+] Lattice size N: ", N)
    println("[+] Time steps T: ", T)
    println("[+] Warm-up steps t0: ", t0)
    println("[+] Integration time step dt: ", dt)
    println("[+] Noise intensity s: ", s)
    println("[+] Diffusion coefficient D: ", D)
    println("[+] HH parameter gNa: ", gNa)
    println("[+] HH parameter gK: ", gK)
    println("[+] HH parameter gL: ", gL)
    println("[+] HH parameter VNa: ", VNa)
    println("[+] HH parameter VK: ", VK)
    println("[+] HH parameter VL: ", VL)
    println("[+] HH parameter C: ", C)
    println("[+] Stimulation current I: ", I)

    # stimulation protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
    stim = [ [[50,350], [1,5], [1,5]],
             [[1900,2200], [25,30], [1,15]] ]
    #stim = [] # empty, no current

    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    blocks = [ [[1,15], [5,10]] ]
    #blocks = [] # empty, no dead areas

    # run simulation
    data = hh2d(N, T, t0, dt, s, D, C, gNa, gK, gL, VNa, VK, VL, I, stim, blocks)
    println("[+] Data dimensions: ", size(data))

    # plot mean voltage
    m = mean(reshape(data, (T,N*N)), dims=2)
    plot(m, "-k"); show()

    # save data
    I_str = rpad(I, 4, '0') # stim. current amplitude as 4-char string
    s_str = rpad(s, 4, '0') # noise as 4-char string
    D_str = rpad(D, 4, '0') # diffusion coefficient as 4-char string
    #fname1 = string("ml2d_I_", I_str, "_s_", s_str, "_D_", D_str, ".npy")
    #npzwrite(fname1, data)
    #println("[+] Data saved as: ", data_filename)

    # video
    fname2 = string("hh2d_I_", I_str, "_s_", s_str, "_D_", D_str, ".mp4")
    #animate_pyplot(fname2, data) # slow
    animate_video(fname2, data) # fast
    println("[+] Animation saved as: ", fname2)
end

main()
