!=========================================================================================
! copyright:
!     Martin van Driel (Martin@vanDriel.de), 2014
!     Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
! license:
!     GNU Lesser General Public License, Version 3 [non-commercial/academic use]
!     (http://www.gnu.org/copyleft/lgpl.html)

module spectral_basis
    use global_parameters, only: sp, dp, pi
    use iso_c_binding, only: c_double, c_int

    implicit none
    private

    public :: lagrange_interpol_2D_td

contains

!== C Wrappers ===========================================================================

!-----------------------------------------------------------------------------------------
subroutine lagrange_interpol_2D_td_wrapped(N, nsamp, points1, points2, coefficients, &
                                           x1, x2, interpolant) &
  bind(c, name="lagrange_interpol_2D_td")

  integer(c_int), intent(in), value  :: N, nsamp
  real(c_double), intent(in)         :: points1(0:N), points2(0:N)
  real(c_double), intent(in)         :: coefficients(1:nsamp, 0:N, 0:N)
  real(c_double), intent(in), value  :: x1, x2
  real(c_double), intent(out)        :: interpolant(nsamp)

  interpolant = lagrange_interpol_2D_td(points1, points2, coefficients, x1, x2)
end subroutine
!-----------------------------------------------------------------------------------------

!== END  C Wrappers ======================================================================

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

end module
!=========================================================================================
