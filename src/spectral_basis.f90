!=========================================================================================
module spectral_basis
    use global_parameters, only: sp, dp, pi
    use iso_c_binding, only: c_double, c_int

    implicit none
    private

    public :: lagrange_interpol_1D_4
    public :: lagrange_interpol_2D_4
    public :: lagrange_interpol_2D_td_4

    public :: lagrange_interpol_1D
    public :: lagrange_interpol_2D
    public :: lagrange_interpol_2D_td

    public :: zelegl
    public :: zemngl2
    public :: def_lagrange_derivs_gll
    public :: def_lagrange_derivs_glj

contains


! --- C Wrapper
subroutine lagrange_interpol_2D_td_wrapped(N, nsamp, points1, points2, coefficients, x1, x2, interpolant) &
  bind(c, name="lagrange_interpol_2D_td")

  integer(c_int), intent(in), value  :: N, nsamp
  real(c_double), intent(in)         :: points1(0:N), points2(0:N)
  real(c_double), intent(in)         :: coefficients(1:nsamp, 0:N, 0:N)
  real(c_double), intent(in), value  :: x1, x2
  real(c_double), intent(out)        :: interpolant(nsamp)

  interpolant = lagrange_interpol_2D_td(points1, points2, coefficients, x1, x2)
end subroutine


!-----------------------------------------------------------------------------------------
subroutine def_lagrange_derivs_gll(npol, G) bind(c, name="def_lagrange_derivs_gll")
!< Defines elemental arrays for the derivatives of Lagrange interpolating
!! Gauss-Lobatto-Legendre (all eta, and xi direction for non-axial elements) or
!! G2(i,j) = \partial_\eta ( l_i(\eta_j) )  i.e. all eta/non-ax xi directions

  integer(c_int), intent(in), value  :: npol
  real(c_double), intent(out)        :: G(0:npol,0:npol)

  real(kind=dp)         :: eta(0:npol)
  real(kind=dp)         :: df(0:npol)
  integer               :: ipol, jpol

  call zelegl(npol, eta)

  do ipol = 0, npol
     call hn_jprime(eta, ipol, npol, df)
     do jpol = 0, npol
        G(ipol, jpol) = df(jpol)
     end do
  end do

end subroutine
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
subroutine def_lagrange_derivs_glj(npol, G0, G2) bind(c, name="def_lagrange_derivs_glj")
!< Defines elemental arrays for the derivatives of Lagrange interpolating
!! Gauss-Lobatto-Jacobi (0,1) points (axial xi direction):
!! G2(i,j) = \partial_\eta ( l_i(\eta_j) )  i.e. all eta/non-ax xi directions

  integer(c_int), intent(in), value      :: npol
  real(c_double), intent(out)            :: G0(0:npol)
  real(c_double), intent(out)            :: G2(0:npol,0:npol)

  real(kind=dp)         :: xi(0:npol)
  real(kind=dp)         :: df(0:npol)
  integer               :: ipol, jpol

  call zemngl2(npol, xi)

  ! axial elements
  do ipol = 0, npol
     call lag_interp_deriv_wgl(df, xi, ipol, npol)
     do jpol = 0, npol
        G2(ipol, jpol) = df(jpol)
     end do
  end do

  G0 = G2(:,0)

end subroutine
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the Lagrangian interpolation polynomial of a function defined by its values at
!  a set of collocation points
function lagrange_interpol_1D(points, coefficients, x)

  real(dp), intent(in)  :: points(0:)
  real(dp), intent(in)  :: coefficients(0:size(points)-1)
  real(dp), intent(in)  :: x
  real(dp)              :: lagrange_interpol_1D
  real(dp)              :: l_j

  integer               :: j, m, n

  n = size(points) - 1
  lagrange_interpol_1D = 0

  !compare: http://en.wikipedia.org/wiki/Lagrange_polynomial#Definition

  do j=0, n
     l_j = 1
     do m=0, n
        if (m == j) cycle
        l_j = l_j * (x - points(m)) / (points(j) - points(m))
     enddo
     lagrange_interpol_1D = lagrange_interpol_1D + coefficients(j) * l_j
  enddo

end function lagrange_interpol_1D
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the Lagrangian interpolation polynomial of a function defined by its values at
!! a set of collocation points in 2D, where the points are a tensorproduct of two sets of
!! points in 1D
function lagrange_interpol_2D(points1, points2, coefficients, x1, x2)

  real(dp), intent(in)  :: points1(0:), points2(0:)
  real(dp), intent(in)  :: coefficients(0:size(points1)-1, 0:size(points2)-1)
  real(dp), intent(in)  :: x1, x2
  real(dp)              :: lagrange_interpol_2D
  real(dp)              :: l_i(0:size(points1)-1), l_j(0:size(points2)-1)

  integer               :: i, j, m, n1, n2

  n1 = size(points1) - 1
  n2 = size(points2) - 1

  do i=0, n1
     l_i(i) = 1
     do m=0, n1
        if (m == i) cycle
        l_i(i) = l_i(i) * (x1 - points1(m)) / (points1(i) - points1(m))
     enddo
  enddo

  do j=0, n2
     l_j(j) = 1
     do m=0, n2
        if (m == j) cycle
        l_j(j) = l_j(j) * (x2 - points2(m)) / (points2(j) - points2(m))
     enddo
  enddo

  lagrange_interpol_2D = 0

  do i=0, n1
     do j=0, n2
        lagrange_interpol_2D = lagrange_interpol_2D  + coefficients(i,j) * l_i(i) * l_j(j)
     enddo
  enddo

end function lagrange_interpol_2D
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the Lagrangian interpolation polynomial of a function defined by its values at
!  a set of collocation points in 2D, where the points are a tensorproduct of two sets of
!  points in 1D, for time dependent coefficients
function lagrange_interpol_2D_td(points1, points2, coefficients, x1, x2)

  real(dp), intent(in)  :: points1(0:), points2(0:)
  real(dp), intent(in)  :: coefficients(:,0:,0:)
  real(dp), intent(in)  :: x1, x2
  real(dp)              :: lagrange_interpol_2D_td(size(coefficients,1))
  real(dp)              :: l_i(0:size(points1)-1), l_j(0:size(points2)-1)

  integer               :: i, j, m1, m2, n1, n2

  n1 = size(points1) - 1
  n2 = size(points2) - 1

  do i=0, n1
     l_i(i) = 1
     do m1=0, n1
        if (m1 == i) cycle
        l_i(i) = l_i(i) * (x1 - points1(m1)) / (points1(i) - points1(m1))
     enddo
  enddo

  do j=0, n2
     l_j(j) = 1
     do m2=0, n2
        if (m2 == j) cycle
        l_j(j) = l_j(j) * (x2 - points2(m2)) / (points2(j) - points2(m2))
     enddo
  enddo

  lagrange_interpol_2D_td(:) = 0

  do i=0, n1
     do j=0, n2
        lagrange_interpol_2D_td(:) = lagrange_interpol_2D_td(:) &
                                     + coefficients(:,i,j) * l_i(i) * l_j(j)
     enddo
  enddo

end function lagrange_interpol_2D_td
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the Lagrangian interpolation polynomial of a function defined by its values at
!! a set of collocation points
!! Special version for polynomial order 4
function lagrange_interpol_1D_4(points, coefficients, x)

  real(dp), intent(in)  :: points(0:4)
  real(dp), intent(in)  :: coefficients(0:4)
  real(dp), intent(in)  :: x
  real(dp)              :: lagrange_interpol_1D_4
  real(dp)              :: l_j

  integer               :: j, m
  integer, parameter    :: n = 4

  lagrange_interpol_1D_4 = 0

  !compare: http://en.wikipedia.org/wiki/Lagrange_polynomial#Definition

  do j=0, n
     l_j = 1
     do m=0, n
        if (m == j) cycle
        l_j = l_j * (x - points(m)) / (points(j) - points(m))
     enddo
     lagrange_interpol_1D_4 = lagrange_interpol_1D_4 + coefficients(j) * l_j
  enddo

end function lagrange_interpol_1D_4
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the Lagrangian interpolation polynomial of a function defined by its values at
!! a set of collocation points in 2D, where the points are a tensorproduct of two sets of
!! points in 1D
!! Special version for polynomial order 4
function lagrange_interpol_2D_4(points1, points2, coefficients, x1, x2)

  real(dp), intent(in)  :: points1(0:4), points2(0:4)
  real(dp), intent(in)  :: coefficients(0:4, 0:4)
  real(dp), intent(in)  :: x1, x2
  real(dp)              :: lagrange_interpol_2D_4
  real(dp)              :: l_i(0:4), l_j(0:4)

  integer               :: i, j, m
  integer, parameter    :: n1 = 4, n2 = 4

  do i=0, n1
     l_i(i) = 1
     do m=0, n1
        if (m == i) cycle
        l_i(i) = l_i(i) * (x1 - points1(m)) / (points1(i) - points1(m))
     enddo
  enddo

  do j=0, n2
     l_j(j) = 1
     do m=0, n2
        if (m == j) cycle
        l_j(j) = l_j(j) * (x2 - points2(m)) / (points2(j) - points2(m))
     enddo
  enddo

  lagrange_interpol_2D_4 = 0

  do i=0, n1
     do j=0, n2
        lagrange_interpol_2D_4 = lagrange_interpol_2D_4  + coefficients(i,j) * l_i(i) * l_j(j)
     enddo
  enddo

end function lagrange_interpol_2D_4
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the Lagrangian interpolation polynomial of a function defined by its values at
!! a set of collocation points in 2D, where the points are a tensorproduct of two sets of
!! points in 1D, for time dependent coefficients
!! Special version for polynomial order 4
function lagrange_interpol_2D_td_4(points1, points2, coefficients, x1, x2)

  real(dp), intent(in)  :: points1(0:4), points2(0:4)
  real(dp), intent(in)  :: coefficients(:,0:,0:)
  real(dp), intent(in)  :: x1, x2
  real(dp)              :: lagrange_interpol_2D_td_4(size(coefficients,1))
  real(dp)              :: l_i(0:4), l_j(0:4)

  integer               :: i, j, m1, m2
  integer, parameter    :: n1 = 4, n2 = 4

  do i=0, n1
     l_i(i) = 1
     do m1=0, n1
        if (m1 == i) cycle
        l_i(i) = l_i(i) * (x1 - points1(m1)) / (points1(i) - points1(m1))
     enddo
  enddo

  do j=0, n2
     l_j(j) = 1
     do m2=0, n2
        if (m2 == j) cycle
        l_j(j) = l_j(j) * (x2 - points2(m2)) / (points2(j) - points2(m2))
     enddo
  enddo

  lagrange_interpol_2D_td_4(:) = 0

  do i=0, n1
     do j=0, n2
        lagrange_interpol_2D_td_4(:) = lagrange_interpol_2D_td_4(:) &
                                     + coefficients(:,i,j) * l_i(i) * l_j(j)
     enddo
  enddo

end function lagrange_interpol_2D_td_4
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> Compute the value of the derivative of the j-th Lagrange polynomial
!! of order N defined by the N+1 GLL points xi evaluated at these very
!! same N+1 GLL points.
subroutine hn_jprime(xi,j,N,dhj)

  real(dp), intent(in)  :: xi(0:N)
  integer,intent(in)    :: j
  integer,intent(in)    :: N
  integer               :: i
  real(dp), intent(out) :: dhj(0:N)
  real(kind=dp)          :: DX,D2X
  real(kind=dp)          :: VN (0:N), QN(0:N)

  dhj(:) = 0d0
  VN(:)= 0d0
  QN(:)= 0d0


  do i = 0, N
     call valepo(N, xi(i), VN(i), DX, D2X)
     if (i == j) QN(i) = 1d0
  end do

  call delegl(N, xi, VN, QN, dhj)

end subroutine hn_jprime
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> Applies more robust formula to return
!! value of the derivative of the i-th Lagrangian interpolant
!! defined over the weighted GLL points computed at these
!! weighted GLL points.
subroutine lag_interp_deriv_wgl(dl,xi,i,N)

  integer, intent(in)    :: N, i
  real(dp), intent(in)   :: xi(0:N)
  real(dp), intent(out)  :: dl(0:N)
  real(kind=dp)          :: mn_xi_i, mnprime_xi_i
  real(kind=dp)          :: mnprimeprime_xi_i
  real(kind=dp)          :: mn_xi_j, mnprime_xi_j
  real(kind=dp)          :: mnprimeprime_xi_j
  real(kind=dp)          :: DN
  integer                :: j

  DN = dble(N)
  call vamnpo(N, xi(i), mn_xi_i, mnprime_xi_i, mnprimeprime_xi_i)

  if ( i == 0) then

     do j = 0, N
        call vamnpo(N, xi(j), mn_xi_j, mnprime_xi_j, mnprimeprime_xi_j)

        if (j == 0) &
           dl(j) = -DN * (DN + 2d0) / 6.d0
        if (j > 0 .and. j < N) &
           dl(j) = 2d0 * ((-1d0)**N) * mn_xi_j / ((1d0 + xi(j)) * (DN + 1d0))
        if (j == N) &
           dl(j) = ((-1d0)**N) / (DN + 1d0)
     end do

  elseif (i == N) then

     do j = 0, N
        call vamnpo(N, xi(j), mn_xi_j, mnprime_xi_j, mnprimeprime_xi_j)
        if (j == 0) &
           dl(j) = ((-1d0)**(N + 1)) * (DN + 1d0) / 4.d0
        if (j > 0 .and. j <  N) &
           dl(j) = -mn_xi_j / (1d0 - xi(j))
        if (j == N) &
           dl(j) = (DN * (DN + 2d0) - 1d0) / 4.d0
     end do

  else

     do j = 0, N
        call vamnpo(N, xi(j), mn_xi_j, mnprime_xi_j, mnprimeprime_xi_j)
        if (j == 0) &
           dl(j) = ( ((-1d0)**(N + 1)) * (DN + 1d0) ) / (2d0 * mn_xi_i * (1d0 + xi(i)))
        if (j > 0 .and. j < N .and. j /= i) &
           dl(j) = ((xi(j) - xi(i))**(-1)) * mn_xi_j / mn_xi_i
        if (j > 0 .and. j < N .and. j == i) &
           dl(j) = - 0.5d0 / (1d0 + xi(j))
        if (j == N) &
           dl(j) = (mn_xi_i * (1d0 - xi(i)))**(-1)
     end do

  end if

end subroutine lag_interp_deriv_wgl
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the nodes relative to the legendre gauss-lobatto formula
subroutine zelegl(n, nodes) bind(c, name="zelegl")

  integer(c_int), intent(in), value  :: n            ! Order of the formula
  real(c_double), intent(out)        :: nodes(0:n)   ! Vector of the nodes
  real(c_double)                     :: sn, x, c, etx, dy, d2y, y
  integer(c_int)                     :: i, n2, it

  if (n == 0) then
     ! does this make sense at all?
     nodes(0) = 0
     return
  endif

  n2 = (n - 1) / 2

  sn = 2 * n - 4 * n2 - 3
  nodes(0) = -1
  nodes(n) =  1

  if (n  ==  1) return

  nodes(n2+1) = 0
  x = 0
  call valepo(n, x, y, dy, d2y)

  if(n == 2) return

  c  = pi / n

  do i=1, n2
     etx = dcos(c * i)
     do it=1, 8
        call valepo(n, etx, y, dy, d2y)
        etx = etx - dy / d2y
     end do
     nodes(i) = -etx
     nodes(n-i) = etx
  end do

end subroutine
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> computes the value of the legendre polynomial of degree n
!! and its first and second derivatives at a given point
subroutine valepo(n, x, y, dy, d2y)

  integer, intent(in)   ::  n   !< degree of the polynomial
  real(dp), intent(in)  ::  x   !< point in which the computation is performed
  real(dp), intent(out) ::  y   !< value of the polynomial in x
  real(dp), intent(out) ::  dy  !< value of the first derivative in x
  real(dp), intent(out) ::  d2y !< value of the second derivative in x
  real(kind=dp)         ::  c1, c2, c4, ym, yp, dym, dyp, d2ym, d2yp
  integer               ::  i

  if (n == 0) then
     y   = 1.d0
     dy  = 0.d0
     d2y = 0.d0

  elseif (n == 1) then
     y   = x
     dy  = 1.d0
     d2y = 0.d0

  else
     y   = x
     dy  = 1.d0
     d2y = 0.d0

     yp   = 1.d0
     dyp  = 0.d0
     d2yp = 0.d0
     do i = 2, n
        c1   = dfloat(i)
        c2   = 2.d0*c1-1.d0
        c4   = c1-1.d0
        ym   = y
        y    = (c2*x*y-c4*yp)/c1
        yp   = ym
        dym  = dy
        dy   = (c2*x*dy-c4*dyp+c2*yp)/c1
        dyp  = dym
        d2ym = d2y
        d2y  = (c2*x*d2y-c4*d2yp+2.d0*c2*dyp)/c1
        d2yp = d2ym
     enddo
  endif

end subroutine valepo
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!>  computes the derivative of a polynomial at the legendre gauss-lobatto
!!  nodes from the values of the polynomial attained at the same points
subroutine delegl(n,et,vn,qn,dqn)

   integer, intent(in)   ::  n        !< the degree of the polynomial
   real(dp), intent(in)  ::  et(0:n)  !< vector of the nodes, et(i), i=0,n
   real(dp), intent(in)  ::  vn(0:n)  !< values of the legendre polynomial at the nodes, vn(i), i=0,n
   real(dp), intent(in)  ::  qn(0:n)  !< values of the polynomial at the nodes, qn(i), i=0,n
   real(dp), intent(out) ::  dqn(0:n) !< derivatives of the polynomial at the nodes, dqz(i), i=0,n
   real(kind=dp)         ::  su, vi, ei, vj, ej, dn, c
   integer               ::  i, j

   dqn(0) = 0.d0
   if (n .eq. 0) return

   do i=0,n
       su = 0.d0
       vi = vn(i)
       ei = et(i)
       do j=0,n
           if (i .eq. j) cycle !goto 2
           vj = vn(j)
           ej = et(j)
           su = su+qn(j)/(vj*(ei-ej))
       enddo !2  continue
       dqn(i) = vi*su
    enddo !1  continue

    dn = dfloat(n)
    c  = .25d0 * dn * (dn+1.d0)
    dqn(0) = dqn(0) - c * qn(0)
    dqn(n) = dqn(n) + c * qn(n)

end subroutine delegl
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!>   Computes the value of the "cylindrical" polynomial
!!   m_n = (l_n + l_{n+1})/(1+x) of degree n
!!   and its first and second derivatives at a given point
!!
!!   implemented after bernardi et al., page 57, eq. (iii.1.10)
subroutine vamnpo(n,x,y,dy,d2y)

  integer, intent(in)   :: n   !< degree of the polynomial
  real(dp), intent(in)  :: x   !< point in which the computation is performed
  real(dp), intent(out) :: y   !< value of the polynomial in x
  real(dp), intent(out) :: dy  !< value of the first derivative in x
  real(dp), intent(out) :: d2y !< value of the second derivative in x
  real(kind=dp)          :: yp, dyp, d2yp, c1
  real(kind=dp)          :: ym, dym, d2ym
  integer               :: i


   y   = 1.d0
   dy  = 0.d0
   d2y = 0.d0
  if (n  ==  0) return

   y   = 1.5d0*x - 0.5d0
   dy  = 1.5d0
   d2y = 0.d0
  if(n  ==  1) return

   yp   = 1.d0
   dyp  = 0.d0
   d2yp = 0.d0
  do i=2,n
      c1 = dble(i-1)
      ym = y
       y = (x-1d0/((2*c1+1d0)*(2*c1+3d0)) ) * y &
          - (c1/(2d0*c1+1d0))*yp
       y = (2d0*c1+3d0)*y/(c1+2d0)
      yp = ym
     dym = dy
      dy = (x-1d0/((2*c1+1d0)*(2*c1+3d0)) ) * dy &
           +yp - (c1/(2d0*c1+1d0))*dyp
      dy = (2d0*c1+3d0)*dy/(c1+2d0)
     dyp = dym
    d2ym = d2y
    d2y  = 2d0*dyp + (x-1d0/((2*c1+1d0)*(2*c1+3d0)) ) * d2y &
           - (c1/(2d0*c1+1d0))*d2yp
    d2y  = (2d0*c1+3d0)*d2y/(c1+2d0)
    d2yp = d2ym
  end do

end subroutine vamnpo
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!>   Computes the nodes relative to the modified legendre gauss-lobatto
!!   FORMULA along the s-axis
!!   Relies on computing the eigenvalues of tridiagonal matrix.
!!   The nodes correspond to the second quadrature formula proposed
!!   by Azaiez et al.
subroutine zemngl2(n, nodes) bind(c, name="zemngl2")

  integer(c_int), intent(in), value  :: n            !< Order of the formula
  real(c_double), intent(out)        :: nodes(0:n) !< vector of the nodes, et(i), i=0,n.
  real(dp), dimension(n-1)           :: d, e
  integer                            :: i, n2

  if (n == 0) then
     nodes(0) = 0
  elseif (n == 1) then
     nodes(0) = -1.d0
     nodes(n) = 1.d0
  elseif (n  ==  2) then
     n2 = (n-1)/2
     nodes(0) = -1.d0
     nodes(n2+1) = 2d-1
     nodes(n) = 1.d0
  elseif (n > 2) then
     ! Form the matrix diagonals and subdiagonals according to
     ! formulae page 109 of Bernardi, Dauge and Maday (Spectral Methods for
     ! axisymmetric domains). MvD: it is page 113 in my copy of the book
     nodes(0) = -1.d0
     nodes(n) = 1.d0

     do i = 1, n-1
        d(i) = 3d0 / (4d0 * (i + 0.5d0) * (i + 1.5d0))
     end do

     do i = 1, n-2
        e(i+1) = dsqrt(i * (i + 3d0)) / (2d0 * (i + 1.5d0))
     end do

     ! Compute eigenvalues
     call tqli(d, e, n-1)

     ! Sort them in increasing order
     call bubblesort(d, e, n-1)

     nodes(1:n-1) = e(1:n-1)

  endif

end subroutine zemngl2
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> This routines returns the eigenvalues of the tridiagonal matrix
!! which diagonal and subdiagonal coefficients are contained in d(1:n) and
!! e(2:n) respectively. e(1) is free. The eigenvalues are returned in array d
subroutine tqli(d,e,n)

  integer, intent(in)             :: n
  real(kind=dp), intent(inout)    :: d(n)
  real(kind=dp), intent(inout)    :: e(n)
  integer                         :: i, iter, l, m
  real(kind=dp)                   :: b, c, dd, f, g, p, r, s

  do i = 2, n
    e(i-1) = e(i)
  end do

  e(n) = 0
  do l=1, n
     iter = 0
     iterate: do
     do m = l, n-1
       dd = abs(d(m)) + abs(d(m+1))
       if (abs(e(m)) + dd .eq. dd) exit
     end do

     if( m == l ) exit iterate
     !if( iter == 30 ) stop 'too many iterations in tqli'
     iter = iter + 1
     g = (d(l+1) - d(l)) / (2. * e(l))
     r = pythag(g, 1d0)
     g = d(m) - d(l) + e(l) / (g + sign(r,g))
     s = 1
     c = 1
     p = 0
     do i = m-1,l,-1
        f      = s * e(i)
        b      = c * e(i)
        r      = pythag(f, g)
        e(i+1) = r
        if(r == 0)then
           d(i+1) = d(i+1) - p
           e(m)   = 0
           cycle iterate
        endif
        s      = f / r
        c      = g / r
        g      = d(i+1) - p
        r      = (d(i) - g) * s + 2. * c * b
        p      = s * r
        d(i+1) = g + p
        g      = c * r - b
     end do
     d(l) = d(l) - p
     e(l) = g
     e(m) = 0
     end do iterate
  end do

end subroutine tqli
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
!> This routine reorders array vin(n) in increasing order and outputs array vout(n).
subroutine bubblesort(vin, vout, n)

  integer, intent(in)            :: n
  real(kind=dp), intent(in)      :: vin(n)
  real(kind=dp), intent(out)     :: vout(n)
  integer                        :: rankmax
  integer                        :: rank(n)
  integer                        :: i, j

  rankmax = 1

  do i = 1, n

     rank(i) = 1

     do j = 1, n
        if((vin(i) > vin(j)) .and. (i /= j)) rank(i) = rank(i) + 1
     end do

     rankmax = max(rank(i), rankmax)
     vout(rank(i)) = vin(i)

  end do

end subroutine bubblesort
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
real(kind=dp) function pythag(a,b)

  real(kind=dp), intent(in) :: a, b
  real(kind=dp)             :: absa, absb

  absa = dabs(a)
  absb = dabs(b)

  if(absa > absb) then
     pythag = absa * sqrt(1. + (absb / absa)**2)
  elseif(absb == 0)then
     pythag = 0
  else
     pythag = absb * sqrt(1. + (absa / absb)**2)
  endif

end function pythag
!-----------------------------------------------------------------------------------------

end module
!=========================================================================================
