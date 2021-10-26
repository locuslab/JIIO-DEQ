# Borrowed from https://github.com/andrenarchy/krypy
# Added features : Added pytorch support and batchified the solvers

import warnings

import numpy
import scipy.linalg

# from modules import utils
import torch
import ipdb

__all__ = ["LinearSystem", "Minres", "Gmres"]


class LinearSystem(object):
    def __init__(
        self,
        A,
        b,
        exact_solution=None,
    ):
        r"""Representation of a (preconditioned) linear system.

        Represents a linear system

        .. math::

          Ax=b

        or a preconditioned linear system

        .. math::

          M M_l A M_r y = M M_l b
          \quad\text{with}\quad x=M_r y.

        :param A: a linear operator on :math:`\mathbb{C}^N` (has to be
          compatible with :py:meth:`~krypy.utils.get_linearoperator`).
        :param b: the right hand side in :math:`\mathbb{C}^N`, i.e.,
          ``b.shape == (N, 1)``.
        :param M: (optional) a self-adjoint and positive definite
          preconditioner, linear operator on :math:`\mathbb{C}^N` with respect
          to the inner product defined by ``ip_B``. This preconditioner changes
          the inner product to
          :math:`\langle x,y\rangle_M = \langle Mx,y\rangle` where
          :math:`\langle \cdot,\cdot\rangle` is the inner product defined by
          the parameter ``ip_B``. Defaults to the identity.
        :param Minv: (optional) the inverse of the preconditioner provided by
          ``M``. This operator is needed, e.g., for orthonormalizing vectors
          for the computation of Ritz vectors in deflated methods.
        :param Ml: (optional) left preconditioner, linear operator on
          :math:`\mathbb{C}^N`. Defaults to the identity.
        :param Mr: (optional) right preconditioner, linear operator on
          :math:`\mathbb{C}^N`. Defaults to the identity.
        :param ip_B: (optional) defines the inner product, see
          :py:meth:`~krypy.utils.inner`.
        :param normal: (bool, optional) Is :math:`M_l A M_r` normal
          in the inner product defined by ``ip_B``? Defaults to ``False``.
        :param self_adjoint: (bool, optional) Is :math:`M_l A M_r` self-adjoint
          in the inner product defined by ``ip_B``? ``self_adjoint=True``
          also sets ``normal=True``. Defaults to ``False``.
        :param positive_definite: (bool, optional) Is :math:`M_l A M_r`
          positive (semi-)definite with respect to the inner product defined by
          ``ip_B``? Defaults to ``False``.
        :param exact_solution: (optional) If an exact solution :math:`x` is
          known, it can be provided as a ``numpy.array`` with
          ``exact_solution.shape == (N,1)``. Then error norms can be computed
          (for debugging or research purposes). Defaults to ``None``.
        """
        self.N = N = b.shape[1]
        self.bsz = b.shape[0]
        """Dimension :math:`N` of the space :math:`\\mathbb{C}^N` where the
        linear system is defined."""
        shape = (self.bsz, N, N)

        # init linear operators
        self.A = A
        self.MlAMr = self.A 

        # process vectors
        self.b, self.exact_solution = b, exact_solution
        self.b_norm = b.norm(dim=-1)

        self.normal = False


    def get_residual(self, z, compute_norm=False):
        r"""Compute residual.

        For a given :math:`z\in\mathbb{C}^N`, the residual

        .. math::

          r = M M_l ( b - A z )

        is computed. If ``compute_norm == True``, then also the absolute
        residual norm

        .. math::

          \| M M_l (b-Az)\|_{M^{-1}}

        is computed.

        :param z: approximate solution with ``z.shape == (N, 1)``.
        :param compute_norm: (bool, optional) pass ``True`` if also the norm
          of the residual should be computed.
        """
        r = self.b - self.A(z)
        if compute_norm:
            return r, r.norm(dim=-1)
        return r

    def get_ip_Minv_B(self):
        """Returns the inner product that is implicitly used with the positive
        definite preconditioner ``M``."""
        return identity (ip_B)

    def __repr__(self):
        ret = "LinearSystem {\n"

        def add(k):
            op = self.__dict__[k]
            if op is not None:
                return "  " + k + ": " + op.__repr__() + "\n"
            return ""

        for k in [
            "A",
            "b",
            "exact_solution",
        ]:
            ret += add(k)
        return ret + "}"



class _KrylovSolver(object):
    """Prototype of a Krylov subspace method for linear systems."""

    def __init__(
        self,
        linear_system,
        x0=None,
        tol=1e-5,
        maxiter=None,
        explicit_residual=False,
        store_arnoldi=False,
        dtype=None,
    ):
        r"""Init standard attributes and perform checks.

        All Krylov subspace solvers in this module are applied to a
        :py:class:`LinearSystem`.  The specific methods may impose further
        restrictions on the operators

        :param linear_system: a :py:class:`LinearSystem`.
        :param x0: (optional) the initial guess to use. Defaults to zero
          vector. Unless you have a good reason to use a nonzero initial guess
          you should use the zero vector, cf. chapter 5.8.3 in *Liesen,
          Strakos. Krylov subspace methods. 2013*. See also
          :py:meth:`~krypy.utils.hegedus`.
        :param tol: (optional) the tolerance for the stopping criterion with
          respect to the relative residual norm:

          .. math::

             \frac{ \| M M_l (b-A (x_0+M_r y_k))\|_{M^{-1}} }
             { \|M M_l b\|_{M^{-1}}}
             \leq \text{tol}

        :param maxiter: (optional) maximum number of iterations. Defaults to N.
        :param explicit_residual: (optional)
          if set to ``False`` (default), the updated residual norm from the
          used method is used in each iteration. If set to ``True``, the
          residual is computed explicitly in each iteration and thus requires
          an additional application of ``M``, ``Ml``, ``A`` and ``Mr`` in each
          iteration.
        :param store_arnoldi: (optional)
          if set to ``True`` then the computed Arnoldi basis and the Hessenberg
          matrix are set as attributes ``V`` and ``H`` on the returned object.
          If ``M`` is not ``None``, then also ``P`` is set where ``V=M*P``.
          Defaults to ``False``. If the method is based on the Lanczos method
          (e.g., :py:class:`Cg` or :py:class:`Minres`), then ``H`` is
          real, symmetric and tridiagonal.
        :param dtype: (optional)
          an optional dtype that is used to determine the dtype for the
          Arnoldi/Lanczos basis and matrix.

        Upon convergence, the instance contains the following attributes:

          * ``xk``: the approximate solution :math:`x_k`.
          * ``resnorms``: relative residual norms of all iterations, see
            parameter ``tol``.
          * ``errnorms``: the error norms of all iterations if
            ``exact_solution`` was provided.
          * ``V``, ``H`` and ``P`` if ``store_arnoldi==True``, see
            ``store_arnoldi``

        If the solver does not converge, a
        :py:class:`~krypy.utils.ConvergenceError` is thrown which can be used
        to examine the misconvergence.
        """
        # sanitize arguments
        if not isinstance(linear_system, LinearSystem):
            raise utils.ArgumentError(
                "linear_system is not an instance of " "LinearSystem"
            )
        self.linear_system = linear_system
        N = linear_system.N
        self.bsz = linear_system.bsz
        self.dtype = linear_system.b.dtype
        self.device = linear_system.b.device
        self.maxiter = N if maxiter is None else maxiter
        self.x0 = x0
        self.explicit_residual = explicit_residual
        self.store_arnoldi = store_arnoldi

        # get initial guess
        self.x0 = self._get_initial_guess(self.x0)

        # get initial residual
        self.r0, self.r0_norm = self._get_initial_residual(self.x0)

        # sanitize initial guess
        if self.x0 is None:
            self.x0 = torch.zeros((self.bsz, N, 1), dtype=self.dtype).to(self.device)
        self.tol = tol

        self.xk = None
        """Approximate solution."""

        # store operator (can be modified in derived classes)
        self.MlAMr = linear_system.MlAMr

        # TODO: reortho
        self.iter = 0
        """Iteration number."""

        self.resnorms = []
        """Relative residual norms as described for parameter ``tol``."""

        # if rhs is exactly(!) zero, return zero solution.
        if self.linear_system.b_norm.norm() == 0:
            self.xk = self.x0 = torch.zeros((self.bsz, N, 1), dtype=self.dtype).to(self.device)
            self.resnorms.append(0.)#torch.zeros(self.bsz, dtype=self.dtype).to(self.device))            # Unsure about dimensions
        else:
            # initial relative residual norm
            self.linear_system.b_norm += 1e-10
            self.resnorms.append((self.r0_norm / (self.linear_system.b_norm)).mean())          # Unsure about dimensions

        # compute error?
        if self.linear_system.exact_solution is not None:
            self.errnorms = []
            """Error norms."""
            self.errnorms.append((self.linear_system.exact_solution - self._get_xk(None)).norm())
        # print(self.r0_norm.norm().item(), 0., self.resnorms[-1].item())
        self._solve()
        self._finalize()

    def _get_initial_guess(self, x0):
        """Get initial guess.

        Can be overridden by derived classes in order to preprocess the
        initial guess.
        """
        return x0

    def _get_initial_residual(self, x0):
        """Compute the residual and its norm.

        See :py:meth:`krypy.linsys.LinearSystem.get_residual` for return
        values.
        """
        return self.linear_system.get_residual(x0, compute_norm=True)

    def _get_xk(self, yk):
        """Compute approximate solution from initial guess and approximate
        solution of the preconditioned linear system."""
        if yk is not None:
            identity = torch.cat([torch.eye((self.x0.shape[1]))]*self.x0.shape[0], dim=0)
            return self.x0 + torch.bmm(identity, yk)
        return self.x0

    def _finalize_iteration(self, yk, resnorm):
        """Compute solution, error norm and residual norm if required.

        :return: the residual norm or ``None``.
        """
        self.xk = None
        # compute error norm if asked for
        if self.linear_system.exact_solution is not None:
            self.xk = self._get_xk(yk)
            self.errnorms.append((self.linear_system.exact_solution - self.xk).norm())

        rkn = None

        # compute explicit residual if asked for or if the updated residual
        # is below the tolerance or if this is the last iteration
        if (
            self.explicit_residual
            or (resnorm / self.linear_system.b_norm).mean() <= self.tol
            or self.iter + 1 == self.maxiter
        ):
            # compute xk if not yet done
            if self.xk is None:
                self.xk = self._get_xk(yk)

            # compute residual norm
            _, rkn = self.linear_system.get_residual(self.xk, compute_norm=True)

            # store relative residual norm
            self.resnorms.append((rkn / self.linear_system.b_norm).mean())
            # print(rkn.norm().item(),self.xk.norm().item(), self.resnorms[-1].item(), "inside")

            # no convergence?
            if self.resnorms[-1] > self.tol:
                # no convergence in last iteration -> raise exception
                # (approximate solution can be obtained from exception)
                if self.iter + 1 == self.maxiter:
                    self._finalize()
                    # raise utils.ConvergenceError(
                    #     (
                    #         "No convergence in last iteration " +
                    #         "(maxiter: {}, ".format(self.maxiter) +
                    #         "residual: {})".format(self.resnorms[-1])
                    #     ),
                    #     self,
                    # )
                # updated residual was below but explicit is not: warn
                # elif (
                #     not self.explicit_residual
                #     and (resnorm / self.linear_system.b_norm).mean() <= self.tol
                # ):
                #     warnings.warn(
                #         "updated residual is below tolerance, explicit "
                #         "residual is NOT! "
                #         "(upd={} <= tol={} < exp={})".format(resnorm.mean(), self.tol, self.resnorms[-1])
                #     )
        else:
            # only store updated residual
            # xk = self._get_xk(yk)
            # _, rkn = self.linear_system.get_residual(xk, compute_norm=True)
            self.resnorms.append((resnorm / self.linear_system.b_norm).mean())
            # print(resnorm.norm().item(),xk.norm().item(), self.resnorms[-1].item(), (rkn/self.linear_system.b_norm).mean().item())

        return rkn

    def _finalize(self):
        pass

    @staticmethod
    def operations(nsteps):
        """Returns the number of operations needed for nsteps of the solver.

        :param nsteps: number of steps.

        :returns: a dictionary with the same keys as the timings parameter.
          Each value is the number of operations of the corresponding type for
          ``nsteps`` iterations of the method.
        """
        raise NotImplementedError(
            "operations() has to be overridden by " "the derived solver class."
        )

    def _solve(self):
        """Abstract method that solves the linear system.
        """
        raise NotImplementedError(
            "_solve has to be overridden by " "the derived solver class."
        )


class Gmres(_KrylovSolver):
    r"""Preconditioned GMRES method.

    The *preconditioned generalized minimal residual method* can be used to
    solve a system of linear algebraic equations. Let the following linear
    algebraic system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y`.
    The preconditioned GMRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \|M M_l(b - A x_k)\|_{M^{-1}} =
      \min_{z \in x_0 + M_r K_k} \|M M_l (b - A z)\|_{M^{-1}}.

    The Arnoldi alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Arnoldi is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is about maxiter+1 vectors for the Arnoldi basis.
    If :math:`M` is used the memory consumption is 2*(maxiter+1).

    If the operator :math:`M_l A M_r` is self-adjoint then consider using
    the MINRES method :py:class:`Minres`.
    """

    def __init__(self, linear_system, ortho="mgs", **kwargs):
        """
        All parameters of :py:class:`_KrylovSolver` are valid in this solver.
        """
        self.ortho = ortho
        super(Gmres, self).__init__(linear_system, **kwargs)

    # def __repr__(self):
    #     string = "krypy GMRES object\n"
    #     string += "    r0 = [{}, ..., {}]\n".format(self.r0[0], self.r0[-1])
    #     string += "    MMlr0_norm = {}\n".format(self.MMlr0_norm)
    #     string += "    MlAMr: {} x {} matrix\n".format(*self.MlAMr.shape)
    #     string += "    Mlr0: [{}, ..., {}]\n".format(self.Mlr0[0], self.Mlr0[-1])
    #     string += "    R: {} x {} matrix\n".format(*self.R.shape)
    #     string += "    V: {} x {} matrix\n".format(*self.V.shape)
    #     string += "    flat_vecs: {}\n".format(self.flat_vecs)
    #     string += "    store_arnoldi: {}\n".format(self.store_arnoldi)
    #     string += "    ortho: {}\n".format(self.ortho)
    #     string += "    tol: {}\n".format(self.tol)
    #     string += "    maxiter: {}\n".format(self.maxiter)
    #     string += "    iter: {}\n".format(self.iter)
    #     string += "    explicit residual: {}\n".format(self.explicit_residual)
    #     string += "    resnorms: [{}, ..., {}]\n".format(
    #         self.resnorms[0], self.resnorms[-1]
    #     )
    #     string += "    x0: [{}, ..., {}]\n".format(self.x0[0], self.x0[-1])
    #     string += "    xk: [{}, ..., {}]".format(self.xk[0], self.xk[-1])
    #     return string

    def _get_xk(self, y):
        if y is None:
            return self.x0
        k = self.arnoldi.iter
        if k > 0:
            yy = torch.triangular_solve(y, self.R[:, :k, :k]).solution
            yk = torch.bmm(self.V[:, :, :k], yy)
            # identity = torch.stack([torch.eye((self.x0.shape[1]))]*self.x0.shape[0], dim=0)
            return self.x0 + yk.squeeze()#torch.bmm(identity, yk)
        return self.x0

    def _solve(self):
        # initialize Arnoldi
        self.arnoldi = Arnoldi(
            self.MlAMr,
            self.r0,
            maxiter=self.maxiter,
            ortho=self.ortho,
            Mv=self.r0,
            Mv_norm=self.r0_norm
        )
        # Givens rotations:
        G = []
        # QR decomposition of Hessenberg matrix via Givens and R
        self.R = torch.zeros([self.bsz, self.maxiter + 1, self.maxiter], dtype=self.dtype).to(self.device)
        y = torch.zeros((self.bsz, self.maxiter + 1, 1), dtype=self.dtype).to(self.device)
        # Right hand side of projected system:
        y[:, 0, 0] = self.r0_norm

        # iterate Arnoldi
        while (
            self.resnorms[-1] > self.tol
            and self.arnoldi.iter < self.arnoldi.maxiter
            and (~self.arnoldi.invariant).sum()
        ):
            k = self.iter = self.arnoldi.iter
            self.arnoldi.advance()

            # Copy new column from Arnoldi
            self.V = self.arnoldi.V
            self.R[:, : k + 2, k] = self.arnoldi.H[:, : k + 2, k]

            # Apply previous Givens rotations.
            for i in range(k):
                self.R[:, i : i + 2, k] = G[i].apply(self.R[:, i : i + 2, k].unsqueeze(-1)).squeeze(-1)

            # Compute and apply new Givens rotation.
            G.append(Givens(self.R[:, k : k + 2, k]))
            self.R[:, k : k + 2, k] = G[k].apply(self.R[:, k : k + 2, k].unsqueeze(-1)).squeeze(-1)
            y[:, k : k + 2] = G[k].apply(y[:, k : k + 2])

            self._finalize_iteration(y[:, : k + 1], y[: , k + 1, 0])

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._get_xk(y[:, : self.arnoldi.iter])

    def _finalize(self):
        super(Gmres, self)._finalize()
        # store arnoldi?
        if self.store_arnoldi:
            self.V, self.H = self.arnoldi.get()

    @staticmethod
    def operations(nsteps):
        """Returns the number of operations needed for nsteps of GMRES"""
        return {
            "A": 1 + nsteps,
            "M": 2 + nsteps,
            "Ml": 2 + nsteps,
            "Mr": 1 + nsteps,
            "ip_B": 2 + nsteps + nsteps * (nsteps + 1) / 2,
            "axpy": 4 + 2 * nsteps + nsteps * (nsteps + 1) / 2,
        }


class Minres(_KrylovSolver):
    r"""Preconditioned MINRES method.

    The *preconditioned minimal residual method* can be used to solve a
    system of linear algebraic equations where the linear operator is
    self-adjoint. Let the following linear algebraic
    system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y` and :math:`M_l A M_r` is self-adjoint with respect
    to the inner product
    :math:`\langle \cdot,\cdot \rangle` defined by ``inner_product``.
    The preconditioned MINRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \|M M_l(b - A x_k)\|_{M^{-1}} =
      \min_{z \in x_0 + M_r K_k} \|M M_l (b - A z)\|_{M^{-1}}.

    The Lanczos alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Lanczos is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is:

    * if ``store_arnoldi==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``store_arnoldi==True``: about maxiter+1 vectors for the Lanczos
      basis.  If :math:`M` is used the memory consumption is 2*(maxiter+1).

    **Caution:** MINRES' convergence may be delayed significantly or even
    stagnate due to round-off errors, cf. chapter 5.9 in [LieS13]_.

    In addition to the attributes described in :py:class:`_KrylovSolver`, the
    following attributes are available in an instance of this solver:

    * ``lanczos``: the Lanczos relation (an instance of :py:class:`Arnoldi`).
    """

    def __init__(self, linear_system, ortho="lanczos", **kwargs):
        """
        All parameters of :py:class:`_KrylovSolver` are valid in this solver.
        Note the restrictions on ``M``, ``Ml``, ``A``, ``Mr`` and ``ip_B``
        above.
        """
        if not linear_system.self_adjoint:
            warnings.warn(
                "Minres applied to a non-self-adjoint "
                "linear system. Consider using Gmres."
            )
        self.ortho = ortho
        super(Minres, self).__init__(linear_system, **kwargs)

    def __repr__(self):
        string = "krypy MINRES object\n"
        string += "    MMlr0 = [{}, ..., {}]\n".format(self.MMlr0[0], self.MMlr0[-1])
        string += "    MMlr0_norm = {}\n".format(self.MMlr0_norm)
        string += "    MlAMr: {} x {} matrix\n".format(*self.MlAMr.shape)
        string += "    Mlr0: [{}, ..., {}]\n".format(self.Mlr0[0], self.Mlr0[-1])
        string += "    flat_vecs: {}\n".format(self.flat_vecs)
        string += "    store_arnoldi: {}\n".format(self.store_arnoldi)
        string += "    ortho: {}\n".format(self.ortho)
        string += "    tol: {}\n".format(self.tol)
        string += "    maxiter: {}\n".format(self.maxiter)
        string += "    iter: {}\n".format(self.iter)
        string += "    explicit residual: {}\n".format(self.explicit_residual)
        string += "    resnorms: [{}, ..., {}]\n".format(
            self.resnorms[0], self.resnorms[-1]
        )
        string += "    x0: [{}, ..., {}]\n".format(self.x0[0], self.x0[-1])
        string += "    xk: [{}, ..., {}]".format(self.xk[0], self.xk[-1])
        return string

    def _solve(self):
        N = self.linear_system.N

        # initialize Lanczos
        self.lanczos = Arnoldi(
            self.MlAMr,
            self.Mlr0,
            maxiter=self.maxiter,
            ortho=self.ortho,
            M=self.linear_system.M,
            Mv=self.MMlr0,
            Mv_norm=self.MMlr0_norm,
            ip_B=self.linear_system.ip_B,
        )

        # Necessary for efficient update of yk:
        W = numpy.column_stack([numpy.zeros(N, dtype=self.dtype), numpy.zeros(N)])
        # some small helpers
        y = [self.MMlr0_norm, 0]  # first entry is (updated) residual
        G2 = None  # old givens rotation
        G1 = None  # even older givens rotation ;)

        # resulting approximation is xk = x0 + Mr*yk
        yk = numpy.zeros((N, 1), dtype=self.dtype)

        # iterate Lanczos
        while (
            self.resnorms[-1] > self.tol
            and self.lanczos.iter < self.lanczos.maxiter
            and not self.lanczos.invariant
        ):
            k = self.iter = self.lanczos.iter
            self.lanczos.advance()
            V, H = self.lanczos.V, self.lanczos.H

            # needed for QR-update:
            R = numpy.zeros((4, 1))  # real because Lanczos matrix is real
            R[1] = H[k - 1, k].real
            if G1 is not None:
                R[:2] = G1.apply(R[:2])

            # (implicit) update of QR-factorization of Lanczos matrix
            R[2:4, 0] = [H[k, k].real, H[k + 1, k].real]
            if G2 is not None:
                R[1:3] = G2.apply(R[1:3])
            G1 = G2
            # compute new givens rotation.
            G2 = Givens(R[2:4])
            R[2] = G2.r
            R[3] = 0.0
            y = G2.apply(y)

            # update solution
            z = (V[:, [k]] - R[0, 0] * W[:, [0]] - R[1, 0] * W[:, [1]]) / R[2, 0]
            W = numpy.column_stack([W[:, [1]], z])
            yk = yk + y[0] * z
            y = [y[1], 0]

            self._finalize_iteration(yk, numpy.abs(y[0]))

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._get_xk(yk)

    def _finalize(self):
        super(Minres, self)._finalize()
        # store arnoldi?
        if self.store_arnoldi:
            if not isinstance(self.linear_system.M, utils.IdentityLinearOperator):
                self.V, self.H, self.P = self.lanczos.get()
            else:
                self.V, self.H = self.lanczos.get()

    @staticmethod
    def operations(nsteps):
        """Returns the number of operations needed for nsteps of MINRES"""
        return {
            "A": 1 + nsteps,
            "M": 2 + nsteps,
            "Ml": 2 + nsteps,
            "Mr": 1 + nsteps,
            "ip_B": 2 + 2 * nsteps,
            "axpy": 4 + 8 * nsteps,
        }



class _RestartedSolver(object):
    """Base class for restarted solvers."""

    def __init__(self, Solver, linear_system, max_restarts=0, **kwargs):
        """
        :param max_restarts: the maximum number of restarts. The maximum
          number of iterations is ``(max_restarts+1)*maxiter``.
        """
        # initial approximation will be set by first run of Solver
        self.xk = None

        # work on own copy of args in order to include proper initial guesses
        kwargs = dict(kwargs)

        # append dummy values for first run
        self.resnorms = [numpy.Inf]
        if linear_system.exact_solution is not None:
            self.errnorms = [numpy.Inf]

        # dummy value, gets reset in the first iteration
        tol = None

        restart = 0
        while restart == 0 or (self.resnorms[-1] > tol and restart <= max_restarts):
            try:
                if self.xk is not None:
                    # use last approximate solution as initial guess
                    kwargs.update({"x0": self.xk})

                # try to solve
                sol = Solver(linear_system, **kwargs)
            except utils.ConvergenceError as e:
                # use solver of exception
                sol = e.solver

            # set last approximate solution
            self.xk = sol.xk
            tol = sol.tol

            # concat resnorms / errnorms
            del self.resnorms[-1]
            self.resnorms += sol.resnorms
            if linear_system.exact_solution is not None:
                del self.errnorms[-1]
                self.errnorms += sol.errnorms

            restart += 1

        if self.resnorms[-1] > tol:
            raise utils.ConvergenceError(
                "No convergence after {} restarts.".format(max_restarts), self
            )


class RestartedGmres(_RestartedSolver):
    """Restarted GMRES method.

    See :py:class:`_RestartedSolver`."""

    def __init__(self, *args, **kwargs):
        super(RestartedGmres, self).__init__(Gmres, *args, **kwargs)




class Arnoldi(object):
    def __init__(
        self, A, v, maxiter=None, ortho="mgs", Mv=None, Mv_norm=None
    ):
        """Arnoldi algorithm.

        Computes V and H such that :math:`AV_n=V_{n+1}\\underline{H}_n`.  If
        the Krylov subspace becomes A-invariant then V and H are truncated such
        that :math:`AV_n = V_n H_n`.

        :param A: a linear operator that can be used with scipy's
          aslinearoperator with ``shape==(N,N)``.
        :param v: the initial vector with ``shape==(N,1)``.
        :param maxiter: (optional) maximal number of iterations. Default: N.
        :param ortho: (optional) orthogonalization algorithm: may be one of

            * ``'mgs'``: modified Gram-Schmidt (default).
            * ``'dmgs'``: double Modified Gram-Schmidt.
            * ``'lanczos'``: Lanczos short recurrence.
            * ``'house'``: Householder.
        :param M: (optional) a self-adjoint and positive definite
          preconditioner. If ``M`` is provided, then also a second basis
          :math:`P_n` is constructed such that :math:`V_n=MP_n`. This is of
          importance in preconditioned methods. ``M`` has to be ``None`` if
          ``ortho=='house'`` (see ``B``).
        :param ip_B: (optional) defines the inner product to use. See
          :py:meth:`inner`.

          ``ip_B`` has to be ``None`` if ``ortho=='house'``. It's unclear to me
          (andrenarchy), how a variant of the Householder QR algorithm can be
          used with a non-Euclidean inner product. Compare
          http://math.stackexchange.com/questions/433644/is-householder-orthogonalization-qr-practicable-for-non-euclidean-inner-products
        """
        N = v.shape[1]
        B = v.shape[0]

        # save parameters
        self.A = A
        self.maxiter = N if maxiter is None else maxiter
        self.ortho = ortho
        self.dtype = v.dtype
        self.device = v.device
        # number of iterations
        self.iter = 0
        # Arnoldi basis
        self.V = torch.zeros((B, N, self.maxiter + 1), dtype=self.dtype).to(self.device)
        # Hessenberg matrix
        self.H = torch.zeros((B, self.maxiter + 1, self.maxiter), dtype=self.dtype).to(self.device)

        if ortho in ["mgs", "dmgs", "lanczos"]:
            self.reorthos = 0
            if ortho == "dmgs":
                self.reorthos = 1
            self.vnorm = Mv_norm
        else:
            raise ArgumentError(
                "Invalid value " + ortho + " for argument 'ortho'. "
                + "Valid are mgs, dmgs and lanczos."
            )
        self.non_invariant = (self.vnorm > 0)
        self.V[self.non_invariant, :, 0] = v[self.non_invariant] / (1e-10 + self.vnorm[self.non_invariant].unsqueeze(-1))
        self.invariant = ~self.non_invariant

    def advance(self):
        """Carry out one iteration of Arnoldi."""
        if self.iter >= self.maxiter:
            raise ArgumentError("Maximum number of iterations reached.")
        # if self.invariant:
        #     raise ArgumentError(
        #         "Krylov subspace was found to be invariant "
        #         "in the previous iteration."
        #     )

        N = self.V.shape[1]
        k = self.iter

        # the matrix-vector multiplication
        Av = self.A(self.V[:, :, k])

        # determine vectors for orthogonalization
        start = 0

        # Lanczos?
        if self.ortho == "lanczos":
            start = k
            if k > 0:
                self.H[:, k - 1, k] = self.H[:, k, k - 1]
                Av -= self.H[:, k, k - 1].unsqueeze(-1) * self.V[:, :, k - 1]

        # (double) modified Gram-Schmidt
        for reortho in range(self.reorthos + 1):
            # orthogonalize
            for j in range(start, k + 1):
                alpha = torch.bmm(self.V[:, :, j].unsqueeze(1), Av.unsqueeze(-1)).squeeze()
                self.H[:, j, k] += alpha
                Av -= alpha.unsqueeze(-1) * self.V[:, :, j]
        self.H[:, k + 1, k] = Av.norm(dim=-1)
        self.invariant = self.H[:, k + 1, k] / (self.H[:, : k + 2, : k + 1].norm(dim=-1).norm(dim=-1) + 1e-10) <= 1e-14
        self.non_invariant = ~self.invariant
        # print(self.non_invariant.sum())
        self.V[self.non_invariant, :, k + 1] = Av[self.non_invariant] / (self.H[self.non_invariant, k + 1, k].unsqueeze(-1) + 1e-10)

        # increase iteration counter
        self.iter += 1

    def get(self):
        k = self.iter
        # if self.invariant:
        #     V, H = self.V[:, :, :k], self.H[:k, :k]
        #     return V, H
        # else:
        V, H = self.V[:, : k + 1], self.H[:, : k + 1, :k]
        return V, H

    def get_last(self):
        k = self.iter
        # if self.invariant:
        #     V, H = None, self.H[:k, [k - 1]]
        #     return V, H
        # else:
        V, H = self.V[:, :, k], self.H[:, : k + 1, [k - 1]]
        return V, H


class Givens:
    def __init__(self, x):
        """Compute Givens rotation for provided vector x.

        Computes Givens rotation
        :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}`
        such that
        :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
        """
        # make sure that x is a vector ;)
        if x.shape[1] != 2:
            raise ArgumentError("x is not a vector of shape (2,1)")

        a = x[:,0]
        b = x[:,1]
        r = torch.sqrt(a**2 + b**2)
        a[r==0.] = 1e-6
        b[r==0.] = 1e-6
        r[r==0.] = 1e-6
        c, s = a/r, b/r
        
        self.c = c
        self.s = s
        self.r = c * a + s * b
        # self.G = numpy.array([[c, s], [-s, c]])
        self.G = torch.stack([torch.stack([c,s], dim=1), torch.stack([-s,c], dim=1)], dim=1)

    def apply(self, x):
        """Apply Givens rotation to vector x."""
        return torch.bmm(self.G, x)