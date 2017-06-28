#! /usr/bin/env python3
#
# Copyright (c) 2017 Joost van Zwieten
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
import math
import matplotlib.colors
import numpy
import scipy.linalg
import re
import operator
import functools
import nutils.function, nutils.topology, nutils.element, nutils.log, nutils.plot, nutils.cache, nutils.cli, nutils.transform, nutils.model


def f_num_rusanov(f, λ, geom):
    # Generate a Rusanov (a.k.a. Local Lax-Friedrichs) numerical flux based on
    # the space-time flux function `f` and the local largest absolute
    # eigenvalue.
    n = nutils.function.normal(geom)
    mean = nutils.function.mean
    jump = nutils.function.jump
    njump = lambda v: jump(v)[...,None]*n[(None,)*v.ndim]
    c = abs(numpy.stack([1, λ]).dot(n))
    return lambda q: mean(f(q)) - c/2*njump(f(q)[:,0])


def burgers(
        *,
        nelems: 'number of elements in spatial dimension' = 16,
        maxrefine: 'maximum number of refinements' = 3,
        degree: 'degree of basis' = 3,
        tol: 'stop tolerance for newton solver' = 1e-6,
        plot_interval: 'time interval for plotting, 0 to disable' = 0.1):
    # Test case with Burgers' equation.  Periodic domain of size 2 pi.  The
    # initial condition is an elevated Gaussian 'bump'.

    # Define the geometry.  `dx` is the width of a coarse element in space and
    # `dt` the width in time.
    width = 2*numpy.pi
    t_end = width
    dx = width / nelems
    dt = dx
    geom = nutils.function.rootcoords(2)*nutils.function.stack([dt, dx])

    # Flux function in space-time (first element refers to time).
    f = lambda q: nutils.function.stack([q[0], q[0]**2/2])[None]
    # Plot functions.
    plot = dict(value=lambda q: q[0])

    # Create a solution iterator.
    sol = solve_slabs_iter(
        nelems=nelems, maxrefine=maxrefine, degree=degree, tol=tol,
        t_end=t_end, refinetol=1e-4, geom=geom, dx=dx, dt=dt,
        f=f,
        f_num=f_num_rusanov(f, 2, geom),
        smoothness_input=lambda q: q[0],
        q0=nutils.function.stack([1+nutils.function.exp(-(geom[1]-numpy.pi)**2*10)]))
    # Plot solution.
    plot_iter(sol, t_end=t_end, plot_interval=plot_interval, plot=plot)


def ifp(
        *,
        nelems: 'number of elements in spatial dimension' = 32,
        maxrefine: 'maximum number of refinements' = 3,
        degree: 'degree of basis' = 3,
        tol: 'stop tolerance for newton solver' = 1e-2,
        plot_interval: 'time interval for plotting, 0 to disable' = 60.,
        t_end: 'end time of simulation' = 5400):
    # IFP test case using the Homogeneous Equilibrium Model.

    width = 10000
    dx = width / nelems
    dt = dx*40*32/10000
    geom = nutils.function.rootcoords(2)*nutils.function.stack([dt, dx])

    # Pipe radius.
    r = 0.146/2 # m
    # Fluid temperature.
    T = 278 # K
    # Reference temperature.
    T_norm = 278 # K
    # Reference gas density.
    ρ_G_norm = 1.26 # kg m**-3
    # Reference pressure.
    p_norm = 1e5 # Pa
    # Dynamic viscosities per phase.
    μ = dict(G=1.8e-5, L=1.516e-3)
    # Pipe roughness.
    pipe_roughness = 1e-8
    # Gravitational acceleration.
    g = 9.8 # m s**-2
    # Pipe inclination.
    φ = 0 # (radians)

    δ2 = nutils.function.eye(2)
    δ3 = nutils.function.eye(3)

    def _ns(q):
        # The unknowns are liquid holdup, scaled pressure and (mixture) velocity.
        α_L, p, u = q
        p *= 1e6
        α = dict(L=α_L, G=1-α_L, M=1)
        del α_L, q

        # Area of 'M': the pipe, 'L': the liquid fraction and 'G': the gas
        # fraction.
        A = {β: numpy.pi*r**2*α[β] for β in 'LGM'}
        # Densities of 'M': the mixture, 'L': the liquid fraction and 'G': the gas fraction.
        ρ = dict(L=1003, G=p*(ρ_G_norm*T_norm/(p_norm*T)))
        ρ['M'] = sum(α[β]*ρ[β] for β in 'LG')
        # Conserved quantities: relative liquid mass, relative gas mass and
        # relative mixture momentum.
        c = nutils.function.stack([α['L']*ρ['L'], α['G']*ρ['G'], α['M']*ρ['M']*u])
        # Space time velocity: time and space.
        U = nutils.function.stack([1, u])

        # Friction.
        μ_m = sum(α[β]*μ[β] for β in 'LG')
        reynolds = ρ['M']*abs(u)*2*r/μ_m
        Θ1 = (-2.457*nutils.function.ln((7/reynolds)**0.9+0.27*pipe_roughness/(2*r)))**16
        Θ2 = (37530/reynolds)**16
        f_wall = 8*((8/reynolds)**12+(Θ1+Θ2)**-1.5)**(1/12)
        τ_wall = -f_wall*ρ['M']*u*abs(u)/(4*r)

        # Space-time flux function.
        f = (c['i']*U['j'] + δ3['i2']*δ2['j1']*α['M']*p).unwrap(geometry=geom)
        # Source term,  only non-zero for the momentum equation.
        source = δ3['i2']*α['M']*(τ_wall-ρ['M']*g*nutils.function.sin(φ))

        # Mass flow rates for 'L': the liquid fraction and 'G': the gas
        # fraction.
        mfr = {β: A[β]*ρ[β]*u for β in 'LG'}

        return locals()

    # Space-time flux function.
    f = lambda q: _ns(q)['f']

    # Plot functions.
    plot = dict(
        liquid_holdup=lambda q: _ns(q)['α']['L'],
        pressure=lambda q: _ns(q)['p'],
        velocity=lambda q: _ns(q)['u'],
        liquid_mass_flow_rate=lambda q: _ns(q)['mfr']['L']
    )

    transition = nutils.function.min(1, nutils.function.max(0, geom[0]/10))
    _bc = dict(
        left=lambda p, mfr, **_: nutils.function.stack([mfr['L']-20, mfr['G']-0.2-0.2*transition, 0]),
        right=lambda p, mfr, **_: nutils.function.stack([(p-1e6)/1e6, 0, 0]),
    )
    bc = dict(
        left=lambda q: _bc['left'](**_ns(q)),
        right=lambda q: _bc['right'](**_ns(q)),
    )

    # Initial state.
    imfr = {'L': 20, 'G': 0.2}
    p_right = (width-geom[1])*1.45e6/width+1e6
    ρ = dict(L=1003, G=p_right*(ρ_G_norm*T_norm/(p_norm*T)))
    α_L = imfr['L']/ρ['L'] / (imfr['G']/ρ['G']+imfr['L']/ρ['L'])
    A = numpy.pi*r**2
    u = imfr['L']/(ρ['L']*A*α_L)
    qguess = q0 = nutils.function.stack([α_L, p_right/1e6, u])
    _bc_steady = dict(
        left=lambda p, mfr, **_: nutils.function.stack([mfr['L']-20, mfr['G']-0.2, 0]),
        right=lambda p, mfr, **_: nutils.function.stack([(p-1e6)/1e6, 0, 0]),
    )
    bc_steady = dict(
        left=lambda q: _bc_steady['left'](**_ns(q)),
        right=lambda q: _bc_steady['right'](**_ns(q)),
    )

    # Create a solution iterator.
    sol = solve_slabs_iter(
        nelems=nelems, maxrefine=maxrefine, degree=degree, tol=tol,
        t_end=t_end, refinetol=1e-4, geom=geom, dx=dx, dt=dt,
        f=f,
        source=lambda q: _ns(q)['source'],
        f_num=f_num_rusanov(f, 1000, geom),
        bc=bc,
        bc_steady=bc_steady,
        smoothness_input=lambda q: q[0],
        qguess=qguess,
        q0='steady state')
    # Plot solution.
    plot_iter(sol, t_end=t_end, plot_interval=plot_interval, plot=plot)


def solve_slabs_iter(
        *,
        nelems, maxrefine, degree, tol, t_end, refinetol=1e-4,
        with_viscosity=True, geom, dx, dt, f, source=None, f_num, bc={},
        bc_steady=None, smoothness_input, q0, qguess=None):

    # `neq`: number of equations.
    neq = len(q0 if not isinstance(q0, str) else qguess)
    # Replace `source` with zeros if not specified (i.e. is `None`).
    source = source or (lambda q: nutils.function.zeros([neq]))
    # Number of viscosity iterations.  Zero implies no viscosity.
    nvisc = 3

    gauss = 'gauss{}'.format

    def stdg_res(slab, target, φ, φ_bc, *, idegree, q_prev=None, ε):

        # Space-time Discontinuous Galerking residual on `slab` with test and
        # trial functions `φ` and, if relevant, basis functions `φ_bc` at the
        # spatial boundaries of `slab`.  The integration is approximated with a
        # Gauss scheme that is exact for polynomials of degree `idegree`.  If
        # `q_prev` is `None`, the residual describes a steady state.
        # Otherwise, `q_prev` is used as the initial condition for this `slab`
        # and should be integrable on the 'past' boundary of `slab`.  The
        # parameter `ε` defines the amount of viscosity.

        # `df`: the derivative of the flux function `f` to unknowns.
        t = nutils.function.DerivativeTarget([neq])
        df = lambda q: nutils.function.replace(t, q, nutils.function.derivative(f(t), t, [0]))

        # `q`: the unknowns.
        q = φ.dot(target)

        ns = dict(f=f(q), source=source(q), fnum=f_num(q))
        ns.update(φ=φ, ε=nutils.function.diagonalize([0]+[1]*(slab.ndims-1))*ε)

        if q_prev is None:
            # Steady state.  TODO: As soon as Nutils has proper tensorial
            # topologies solve this on the spatial domain only.
            _Integral = functools.partial(Integral, degree=idegree, geometry=geom, ns=ns)
            res = (
                  _Integral('-φ_ni,1 f_i1 + ε_11 φ_ni,1 f_i0,1 - φ_ni source_i', domain=slab)
                + _Integral('-[φ_ni] fnum_i1 n_1 + [φ_ni] {ε_11 f_i0,1} n_1 - {ε_11 φ_ni,1} [f_i0] n_1', domain=slab.interfaces)
            )
        else:
            ns['fprev'] = f(q_prev)
            _Integral = functools.partial(Integral, degree=idegree, geometry=geom, ns=ns)
            res = (
                  _Integral('-φ_ni,j f_ij + ε_jk φ_ni,j f_i0,k - φ_ni source_i', domain=slab)
                + _Integral('φ_ni f_ij n_j', domain=slab.boundary['future'])
                + _Integral('φ_ni fprev_ij n_j', domain=slab.boundary['past'])
                + _Integral('-[φ_ni] fnum_ij n_j + [φ_ni] {ε_jk f_i0,k} n_j - {ε_jk φ_ni,j} [f_i0] n_k', domain=slab.interfaces)
            )

        # Boundaries.  At the boundary there are additional basis functions
        # `φbc` and unknowns `qbc`.  The system is transformed into a set of
        # scalar equations and the transformed unknowns `qbc` are constrained
        # to the transformed unknowns inside the slab if the scalar velocity is
        # directed outward.  Otherwise the (possibly nonlinear) boundary
        # conditions `bc` are applied.  The scalar systems are sorted
        # algebracially on the eigenvalues, inward first.  The first `n` boundary
        # conditions `bc` are used, where `n` is the number of inward eigenvalues.

        # Find linearization matrices `A` and `B` of this system at the boundary.
        A = df(q).dot(nutils.function.normal(geom), axis=1)
        B = df(q)[:,0]
        # The `mask` and complementary `cmask` indicate outward or inward waves.
        mask = nutils.function.diagonalize(0.5+0.5*nutils.function.sign(eigvalreal(A, B)))
        cmask = nutils.function.diagonalize(0.5-0.5*nutils.function.sign(eigvalreal(A, B)))
        for side in sorted(bc_steady if q_prev is None else bc):
            q_bc = φ_bc[side].dot(target)
            # `RI`: inverse of matrix of right eigenvectors of the generalized
            # eigenvalue problem `A x = λ B x`.
            RI = nutils.function.inverse(eigrvecreal(A, B))
            RI /= nutils.function.norm2(RI, axis=-1)[..., None]
            nsside = dict(φ=φ, φbc=φ_bc[side], fbc=f(q_bc), q=q, qbc=q_bc, bc=bc[side](q_bc), RI=RI, mask=mask, cmask=cmask)
            _Integral = functools.partial(Integral, degree=idegree, geometry=geom, ns=nsside)
            res += _Integral('φ_ni fbc_ij n_j + φbc_ni (mask_ij (RI_jk (qbc_k - q_k)) + cmask_ij bc_j)', domain=slab.boundary[side])

        return res

    # Define the first slab.  Currently only one spatial dimension.
    slab = SlabTopology.first(nelems=nelems, periodic=not bc)

    # `ε_factor`: scaling factor for the amount of viscosity at the coarse
    # mesh, scales with element width `dx`.
    ε_factor0 = dx*1e-2

    sides = tuple(sorted(bc))

    # Determine the initial solution.
    if q0 == 'steady state':
        # Find the steady state solution of this system, subject to the
        # boundary conditions `bc_steady`.

        # Create a basis `φ` on the slab and `φ_bc` on the boundaries, if any.
        φ = slab.basis('discont', degree=degree).vector(neq)
        if sides:
            φ, *φ_bc = nutils.function.chain([φ]+[slab.boundary[side].basis('discont', degree=degree).vector(neq) for side in sides])
            φ_bc = dict(zip(sides, φ_bc))
        target = nutils.function.DerivativeTarget([len(φ)])

        # Determine an initial solution by projecting the supplied `qguess`
        # onto the basis `φ`.
        lhs0 = slab.project(qguess, onto=φ, ischeme=gauss(degree*4), geometry=geom)
        for side in sides:
            lhs0 |= slab.boundary[side].project(qguess, onto=φ_bc[side], ischeme=gauss(degree*4), geometry=geom)

        # Create a steady state residual, without viscosity.
        res = stdg_res(slab, target, φ, φ_bc, ε=0, idegree=4*degree)

        # Find an approximate solution using Newton and let this be the initial
        # solution.
        with nutils.log.context('newton'):
            sol = nutils.model.newton(target, res, lhs0=lhs0).solve(tol=tol, maxiter=100)
        q_prev = φ.dot(sol)
    else:
        # Use the supplied initial sulution.
        q_prev = q0

    iargs = dict(ischeme=gauss(4*degree), geometry=geom)

    # Keep track of the actual number of elements in all slabs until `t_end`
    # and the number of elements that would have been required on the
    # `maxrefine` times structured refinements of the coarse slabs.
    totalnelems = 0
    totalmaxnelems = 0

    # Loop over all slabs and obtain solutions at each slab.
    for i in nutils.log.range('slab', math.ceil(t_end / dt)):

        # Per slab do the following:
        # * [refine]  If we are not yet at the finest level, obtain a solution
        #   at the current topology without viscosity, determine which elements
        #   need to be refined and either apply the refinement or, if no
        #   elements need to be refined, advance to the next slab.
        # * [viscosity]  If we are at the finest level, repeatedly obtain a
        #   solution with viscosity.  At the first iteration the viscosity is
        #   constant.  At subsequent iterations the viscosity is updated
        #   locally based on the roughness.

        for j in nutils.log.range('iter', maxrefine+1+nvisc):

            if j <= maxrefine:
                # Create new bases for the new topology (happens in all refine
                # iterations and the first viscosity iteration).  In subsequent
                # iterations we keep the bases, because the topology doesn't
                # change anymore.  The bases are
                #   `φ`:         a polynomial basis of degree `degree`,
                #   `φ_lowpass`: a polynomial basis of degree `degree-1,
                #   `ψ`:         a degree 0 basis and
                #   `φ_bc`:      a polynomial basis of degree `degree` at the
                #                spatial boundaries of the slab.
                φ = slab.basis('discont', degree=degree).vector(neq)
                φ_lowpass = slab.basis('discont', degree=degree-1)
                ψ = slab.basis('discont', degree=0)
                φ_bc = [slab.boundary[side].basis('discont', degree=degree).vector(neq) for side in sides]
                φ, *φ_bc = nutils.function.chain([φ]+φ_bc)
                φ_bc = dict(zip(sides, φ_bc))

                # Create an initial guess for the solution on this topology.  If
                # this is the first iteration of this slab, use `qguess` if given
                # and zeros otherwise.  If this is not the first iteration, use the
                # solution obtained at the previous iteration as initial guess.
                # The initial guess and the solution are stored in one and the same
                # variable: `lhs`.
                with nutils.log.context('initial guess'):
                    if j > 0:
                        # This is not the first iteration.  Project the
                        # previous solution onto the new basis.
                        lhs0 = slab.project(q, onto=φ, **iargs)
                        for side in sorted(bc):
                            lhs0 |= slab.boundary[side].project(q_bc[side], onto=φ_bc[side], **iargs)
                        lhs = lhs0
                    elif qguess is not None:
                        # This is the first iteration (of this slab) and there
                        # is a `qguess`.  Project `qguess` onto the basis.
                        lhs0 = slab.project(qguess, onto=φ, **iargs)
                        for side in sorted(bc):
                            lhs0 |= slab.boundary[side].project(qguess, onto=φ_bc[side], **iargs)
                        lhs = lhs0
                    else:
                        # This is the first iteration and there is not `qguess`
                        # supplied.
                        lhs = None

                # Set viscosity to zero.
                ε_coeffs = numpy.zeros(ψ.shape)
                ε = 0
            else:
                # Define viscosity function.  The viscosity is constant per element.
                ε = ψ.dot(ε_coeffs)

            # Obtain a solution on this topology using Newton's method.
            target = nutils.function.DerivativeTarget([len(φ)])
            res = stdg_res(slab, target, φ, φ_bc, ε=ε, q_prev=q_prev, idegree=4*degree)
            with nutils.log.context('newton'):
                lhs = nutils.model.newton(target, res, lhs0=lhs).solve(tol=tol, maxiter=100)

            # Create solution functions given the coefficient vector `lhs`.
            # `q` is the solution on this slab and `q_bc` is the solution at
            # the spatial boundaries.
            q = φ.dot(lhs)
            q_bc = {side: φ_bc[side].dot(lhs) for side in sorted(bc)}

            # Compute the roughness, or inverse smoothness, of the solution `q`
            # by comparing per element the high-pass filtered solution with the
            # unfiltered solution.
            w = smoothness_input(q)
            w_lowpass = slab.projection(w, onto=φ_lowpass, **iargs)
            roughness_sqr = operator.truediv(*slab.project([(w - w_lowpass)**2, w**2], onto=ψ.vector(2), **iargs).reshape(2, len(ψ)))

            # If the roughness is below a threshold for all elements and the
            # viscosity is zero everywhere we accept the current solution.
            if (ε_coeffs <= 0).all() and (roughness_sqr <= refinetol**2).all():
                nutils.log.info('no further refinement or viscosity needed')
                break

            if j < maxrefine:
                # Refine elements where the roughness is too large and refine
                # neighbors.  Furthermore, make sure that to neighboring
                # elements differ at most one in refinement level.
                mask = roughness_sqr > refinetol**2
                refine = set()
                add_refine = {elem.transform for elem in slab.supp(ψ, mask)}
                while add_refine:
                    refine |= set(add_refine)
                    add_refine = slab.find_larger_neighbors(add_refine)
                assert len(refine) > 0
                slab = slab.refined_by(refine)
                nutils.log.user('refined {} elements'.format(len(refine)) if len(refine) != 1 else 'refined 1 element')
            elif j < maxrefine+nvisc:
                # Update the viscosity coefficients based on the smoothness
                # indicator and clip negative values to zero.
                ε_coeffs += (numpy.log(roughness_sqr)-2*numpy.log(refinetol))*ε_factor0*2**-maxrefine
                ε_coeffs[ε_coeffs < 0] = 0
                nutils.log.user('updated viscosity')

        # Update the stats.
        totalnelems += len(slab)
        totalmaxnelems += nelems*2**(slab.ndims*maxrefine)
        nutils.log.user('{}/{} ({:.1f}%) elements in this slab'.format(len(slab), nelems*2**(slab.ndims*maxrefine), 100*len(slab)/(nelems*2**(slab.ndims*maxrefine))))

        yield slab, i*dt, (i+1)*dt, geom, φ, q, ε, ψ.dot(roughness_sqr**0.5)

        # Move on to the next slab and update the initial condition `q_prev` of
        # the new slab.
        slab = slab.advance()
        q_prev = nutils.function.opposite(q)

    nutils.log.user('{}/{} ({:.1f}%) elements'.format(totalnelems, totalmaxnelems, 100*totalnelems/totalmaxnelems))


def plot_iter(sol, *, t_end, plot, plot_interval):

    v_names = sorted(plot)
    p = [v for n, v in sorted(plot.items())]

    x = []
    u = []
    v = [[] for n in v_names]
    plot_visc = []
    plot_elem_density = []

    last_time_plot_index = -1

    s_plot_int = str(plot_interval)
    if s_plot_int.endswith('.0'):
        s_plot_int = s_plot_int[:-2]
    t_width = len(str(t_end).split('.')[0])
    if '.' in s_plot_int:
        n_decimals = len(s_plot_int.split('.', 1)[1])
        t_width += n_decimals + 1
    else:
        n_decimals = 0
    t_fmt = 't = {{:{}.{}f}}'.format(t_width, n_decimals)

    for slab, t_past, t_future, geom, φ, q, ε, r in sol:

        with nutils.log.context('plot'):

            ψ = slab.basis('discont', degree=0)
            dens = ψ.dot(1 / slab.integrate(ψ, ischeme='gauss1', geometry=geom))
            x_i, u_i, visc_i, dens_i, *V_i = slab.elem_eval([geom[[1,0]], r, ε, dens, *(p_i(q) for p_i in p)], ischeme='bezier5', separate=True)
            x += x_i
            u += u_i
            plot_visc += visc_i
            plot_elem_density += dens_i
            for _v, v_i in zip(v, V_i):
                _v += v_i

            while plot_interval and (last_time_plot_index+1) * plot_interval <= t_future:
                last_time_plot_index += 1
                s = slab.timeslice(last_time_plot_index*plot_interval/(t_future-t_past))
                x_s, *V_s = s.elem_eval([geom[1], *(p_i(q) for p_i in p)], ischeme='bezier9', separate=True)
                for name, v_s in zip(v_names, V_s):
                    with nutils.plot.PyPlot('solution_{}'.format(name)) as plt:
                        plt.figtext(0.025, 0.975, t_fmt.format(last_time_plot_index*plot_interval), family='monospace', horizontalalignment='left', verticalalignment='top')
                        plt.title(name.replace('_', ' '))
                        plt.plot(numpy.array(x_s).T, numpy.array(v_s).T, 'k')

    with nutils.log.context('plot'):

        for name, v_i in zip(v_names, v):
            with nutils.plot.PyPlot('space_time_solution_{}'.format(name)) as plt:
                plt.mesh(x, v_i, aspect='auto', edgecolors='none')
                plt.colorbar()

        with nutils.plot.PyPlot('roughness') as plt:
            plt.mesh(x, u, aspect='auto', edgecolors='none', norm=matplotlib.colors.LogNorm())
            plt.colorbar()

        with nutils.plot.PyPlot('viscosity') as plt:
            plt.mesh(x, plot_visc, aspect='auto', edgecolors='none')
            plt.colorbar()

        with nutils.plot.PyPlot('elem_density') as plt:
            plt.mesh(x, plot_elem_density, aspect='auto', edgecolors='none')
            plt.colorbar()

        with nutils.plot.PyPlot('mesh') as plt:
            plt.mesh(x, aspect='auto', edgecolors='k')


class SlabTopology(nutils.topology.Topology):
    # Helper class for creating a sequence of space-time slabs, supporting
    # local refinement and integration of functions living on two subsequent
    # slabs at the interface.

    @classmethod
    def first(cls, *, nelems, periodic=False):
        return cls(nelems, periodic, 0, None, None)

    def advance(self):
        return type(self)(self._nelems, self._periodic, self._index+1, None, self._topo)

    def __init__(self, nelems, periodic, index, topo, previous_topo):

        self._nelems = nelems
        self._periodic = periodic
        self._index = index
        self._previous_topo = previous_topo
        if topo is None:
            shape = 1, nelems
            wrap = 0, nelems if periodic else 0
            root = nutils.transform.roottrans('spacetime', wrap)
            taxis = nutils.topology.DimAxis(index, index+1, isperiodic=False)
            saxis = nutils.topology.DimAxis(0, nelems, isperiodic=periodic)
            bnames = ['past', 'future']
            if not periodic:
                bnames += ['left', 'right']
            self._topo = nutils.topology.StructuredTopology(root, [taxis, saxis], bnames=bnames)
        else:
            self._topo = topo

        super().__init__(ndims=2)

    def __iter__(self):
        return iter(self._topo)

    def __len__(self):
        return len(self._topo)

    @nutils.cache.property
    def elements(self):
        return tuple(self)

    @property
    def periodic(self):
        return False, self._periodic

    @nutils.cache.property
    def boundary(self):
        oldbtopo = self._topo.boundary

        past = oldbtopo['past']
        if self._previous_topo and not isinstance( self._previous_topo.boundary['future'], nutils.topology.StructuredTopology ):
            previous_future = self._previous_topo.boundary['future'].edict
            while True:
                elems = [elem for elem in past if not elem.opposite.lookup(previous_future)]
                if not elems:
                    break
                past = past.refined_by(elems)

        btopos = [past, oldbtopo['future']]
        names = ['past', 'future']
        if not self._periodic:
            btopos += [oldbtopo['left'], oldbtopo['right']]
            names += ['left', 'right']
        return nutils.topology.UnionTopology(btopos, names)

    @property
    def interfaces(self):
        return self._topo.interfaces

    def basis(*args, **kwargs):
        self, *args = args
        return self._topo.basis(*args, **kwargs)

    @property
    def basetopo(self):
        if isinstance( self._topo, nutils.topology.StructuredTopology ):
            return self._topo
        else:
            return self._topo.basetopo

    @property
    def refined(self):
        topo = self._topo.refined
        return type(self)(self._nelems, self._periodic, self._index, topo, self._previous_topo)

    def refined_by(self, refine):
        topo = self._topo.refined_by(refine)
        return type(self)(self._nelems, self._periodic, self._index, topo, self._previous_topo)

    def __str__(self):
        return '<SlabTopology {} nelems={}{}>'.format(self._index, self._nelems, ' periodic' if self._periodic else '')

    @nutils.log.title
    def timeslice(self, t):
        if t <= self._index:
            return self._topo.boundary['past']
        elif t >= self._index+1:
            return self._topo.boundary['future']
        else:
            elems = []
            for elem in self:
                offset = nutils.transform.offset(elem.transform[1:])[0]
                linear = nutils.transform.linear(elem.transform[1:])
                if offset <= t < offset + linear:
                    trans = elem.transform << nutils.transform.affine(linear=numpy.eye(2)[:,1:], offset=[(t-offset)/linear, 0], isflipped=False)
                    elems.append(nutils.element.Element(elem.edges[0].reference, trans, trans, oriented=True))
            return nutils.topology.UnstructuredTopology(1, elems)

    @nutils.cache.property
    def elemlevels(self):
        if isinstance(self._topo, nutils.topology.StructuredTopology):
            return numpy.zeros([len(self)])
        base = len(next(iter(self._topo.basetopo)).transform)
        l = numpy.empty([len(self)], dtype=int)
        for i, elem in enumerate(self):
            l[i] = len(elem.transform) - base
        return l

    @nutils.cache.property
    def trans_index_level(self):
        # Build a lookup table for level and element indices given elements in this
        # topology.
        return {
          elem.transform: (ielem, ilevel)
          for ilevel, level in enumerate( self._topo.levels )
          for ielem, elem in enumerate( level )
        }

    def find_larger_neighbors(self, transforms):
        if isinstance(self._topo, nutils.topology.StructuredTopology):
            return []
        trans_index_level = self.trans_index_level
        larger = []
        edict = self._topo.edict
        interfaces = []
        for trans in transforms:
          # Get `level`, element number at `level` of `trans`.
          itrans, ilevel = trans_index_level[trans]
          level = self._topo.levels[ilevel]
          # Loop over neighbours of `trans`.
          for itransedge, ineighbor in enumerate( level.connectivity[itrans] ):
            if ineighbor < 0:
              # Not an interface.
              continue
            neighbor = level.elements[ineighbor].transform
            # Lookup `neighbor` (from the same `level` as `trans`) in this topology.
            head, tail = neighbor.lookup( edict ) or (None, None)
            if tail:
                # A parent of `neighbor` exists in this topology.
                larger.append(head)
        return larger


def eval_expr(s, **ns):
    # Temporary solution until Nutils has namespaces and a new expression
    # parser.

    # Add whitespace around arithmetic operators and parentheses.
    s = re.sub('[+-]|\\(|\\)|\\{|\\}|\\[|\\]', lambda m: ' '+m.group(0)+' ', s)
    # Remove repeated spaces.
    s = re.sub('\\ +', lambda m: ' ', s).strip()
    # Replace mean.
    s = re.sub('\\{(.*?)\\}', lambda t: 'mean({})'.format(t.group(1)), s)
    # Replace jump.
    s = re.sub('\\[(.*?)\\]', lambda t: 'jump({})'.format(t.group(1)), s)
    # Tokenize.
    s = s.split()
    # Add multiplication signs between two consecutive expressions.
    for i in reversed(range(len(s)-1)):
        if s[i] not in '+-' and not s[i].endswith('(') and s[i+1] not in '+-)':
            s.insert(i+1, '*')
    # Replace subscripts.
    s = ['{}["{}"]'.format(*t.split('_', 1)) if '_' in t else t for t in s]
    # Stitch.
    s = ' '.join(s)
    return eval(s, dict(mean=nutils.function.mean, jump=nutils.function.jump, n=nutils.function.normal, **ns))


def Integral(expr, *, domain, degree, geometry, ns):
    # `nutils.model.Integral` wrapper supporting tensor expressions and a
    # namespace.

    if isinstance(expr, str):
        expr = eval_expr(expr, **ns)
    return nutils.model.Integral(expr, domain=domain, degree=degree, geometry=geometry)


class EigReal(nutils.function.Array):
    # Computes eigenvalues and eigenvectors assuming both are real-valued.  The
    # derivative is approximated with zero.

    def __init__(self, A, B, ret):
        assert ret in {'val', 'rvec'}
        assert A.dtype != complex and B.dtype != complex
        self.A, self.B = nutils.function._matchndim(A, B)
        self.ret = ret
        shape = nutils.function._jointshape(self.A.shape, self.B.shape)
        if self.ret == 'val':
            shape = shape[:-1]
        super().__init__(args=[self.A, self.B], shape=shape, dtype=float)

    def evalf(self, A, B):
        assert A.ndim == self.A.ndim + 1
        assert B.ndim == self.B.ndim + 1
        ishape = shape = nutils.function._jointshape(A.shape, B.shape)
        if self.ret == 'val':
            shape = shape[:-1]
        result = numpy.zeros(shape, float)
        for i in numpy.ndindex(*ishape[:-2]):
            iA = tuple(0 if k == 1 else j for j, k in zip(i, A.shape))
            iB = tuple(0 if k == 1 else j for j, k in zip(i, B.shape))
            M = numpy.dot(numpy.linalg.inv(B[iB]), A[iA])
            val, vec = scipy.linalg.eig(M)
            j = numpy.argsort(val.real)
            if self.ret == 'val':
                result[i] = val.real[j]
            elif self.ret in {'rvec', 'lvec'}:
                result[i] = vec.real[:,j]
                #for k in range(len(result[i])):
                #    result[i][:,k] *= numpy.sign(result[i][numpy.argmax(abs(result[i][:,k])),k])
        return result

    def _derivative(self, var, axes, seen):
        return nutils.function.derivative(nutils.function.zeros(self.shape), var, axes, seen)

    def _edit(self, op):
        return _eigreal(op(self.A), op(self.B), self.ret)


def _eigreal(A, B, ret):
    return EigReal(A, B, ret)


def eigvalreal(A, B):
    return _eigreal(A, B, ret='val')

def eiglvecreal(A, B):
    return _eigreal(A, B, ret='lvec')

def eigrvecreal(A, B):
    return _eigreal(A, B, ret='rvec')


if __name__ == '__main__':
    nutils.cli.choose(burgers, ifp)
