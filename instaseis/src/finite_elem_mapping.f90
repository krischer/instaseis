!=========================================================================================
! this module provides the functionality to map physical to reference coordinates and back
! note that the definition of the Jacobian matrix is
!
! copyright:
!     Martin van Driel (Martin@vanDriel.de), 2014
!     Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
! license:
!     GNU Lesser General Public License, Version 3 [non-commercial/academic use]
!     (http://www.gnu.org/copyleft/lgpl.html)
!
!     | ds / dxi  ds / deta |
! J = |                     |
!     | dz / dxi  dz / deta |
!
! and its inverse is
!
!        | dxi  / ds  dxi  / dz |
! J^-1 = |                      |
!        | deta / ds  deta / dz |
!
! which means that the gradient of a function u transforms as
!
! | d_xi  u |        | d_s u |
! |         | = J^T  |       |
! | d_eta u |        | d_z u |
!
! and
!
! | d_s u |            | d_xi  u |
! |       | = (J^-1)^T |         |
! | d_z u |            | d_eta u |
!
! node numbering of the elements is
!
! 4 - - - - - - - 3
! |       ^       |
! |   eta |       |
! |       |       |
! |        --->   |
! |        xi     |
! |               |
! |               |
! 1 - - - - - - - 2
!
! for axial elements, xi is hence the normal and eta the parallel direction to the axis

module finite_elem_mapping
    use global_parameters, only            : sp, dp, pi
    use iso_c_binding, only: c_double, c_int, c_bool

    implicit none
    private

    public  :: inside_element

    public  :: mapping
    public  :: inv_mapping
    public  :: jacobian
    public  :: inv_jacobian
    public  :: jacobian_det

    public  :: mapping_semino
    public  :: inv_mapping_semino
    public  :: jacobian_semino
    public  :: inv_jacobian_semino

    public  :: mapping_semiso
    public  :: inv_mapping_semiso
    public  :: jacobian_semiso
    public  :: inv_jacobian_semiso

    public  :: mapping_spheroid
    public  :: inv_mapping_spheroid
    public  :: jacobian_spheroid
    public  :: inv_jacobian_spheroid

    public  :: mapping_subpar
    public  :: inv_mapping_subpar
    public  :: jacobian_subpar
    public  :: inv_jacobian_subpar
contains

!-----------------------------------------------------------------------------------------
subroutine inside_element(s, z, nodes, element_type, tolerance, in_element, xi, eta) &
    bind(c, name="inside_element")
!< test whether a point described by global coordinates s,z is inside an element
!< and return reference coordinates xi and eta

  real(c_double), intent(in), value             :: s, z
  real(c_double), intent(in)                    :: nodes(4,2)
  integer(c_int), intent(in), value             :: element_type
  real(c_double), intent(in), value             :: tolerance
  logical(c_bool), intent(out)                  :: in_element
  real(c_double), intent(out)                   :: xi, eta
  real(c_double)                                :: tolerance_loc
  real(c_double)                                :: inv_mapping(2)

  tolerance_loc = tolerance

  select case(element_type)
     case(0)
        inv_mapping = inv_mapping_spheroid(s, z, nodes)
     case(1)
        inv_mapping = inv_mapping_subpar(s, z, nodes)
     case(2)
        inv_mapping = inv_mapping_semino(s, z, nodes)
     case(3)
        inv_mapping = inv_mapping_semiso(s, z, nodes)
     case default
        write(6,*) 'ERROR: unknown element type: ', element_type
        stop
  end select

  in_element = (inv_mapping(1) >= -1 - tolerance_loc .and. &
                inv_mapping(1) <=  1 + tolerance_loc .and. &
                inv_mapping(2) >= -1 - tolerance_loc .and. &
                inv_mapping(2) <=  1 + tolerance_loc)

  xi = inv_mapping(1)
  eta = inv_mapping(2)

end subroutine inside_element
!-----------------------------------------------------------------------------------------

!!!!!!! WRAPPING ROUTINES FOR MAPPING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-----------------------------------------------------------------------------------------
function inv_mapping(s, z, nodes, element_type)
!< This routines computes the coordinates in the reference coordinates xi, eta
!! this is nonlinear, but as the forward mapping is smooth, the gradient search
!! converges very quickly

  real(kind=dp), intent(in) :: s, z, nodes(4,2)
  integer, intent(in)       :: element_type
  real(kind=dp)             :: inv_mapping(2)

  select case(element_type)
     case(0)
        inv_mapping = inv_mapping_spheroid(s, z, nodes)
     case(1)
        inv_mapping = inv_mapping_subpar(s, z, nodes)
     case(2)
        inv_mapping = inv_mapping_semino(s, z, nodes)
     case(3)
        inv_mapping = inv_mapping_semiso(s, z, nodes)
     case default
        write(6,*) 'ERROR: unknown element type: ', element_type
        stop
  end select

end function inv_mapping
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function mapping(xi, eta, nodes, element_type)

  real(kind=dp), intent(in) :: xi, eta, nodes(4,2)
  integer, intent(in)       :: element_type
  real(kind=dp)             :: mapping(2)

  select case(element_type)
     case(0)
        mapping = mapping_spheroid(xi, eta, nodes)
     case(1)
        mapping = mapping_subpar(xi, eta, nodes)
     case(2)
        mapping = mapping_semino(xi, eta, nodes)
     case(3)
        mapping = mapping_semiso(xi, eta, nodes)
     case default
        write(6,*) 'ERROR: unknown element type: ', element_type
        stop
  end select

end function mapping
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function jacobian(xi, eta, nodes, element_type)

  real(kind=dp), intent(in) :: xi, eta, nodes(4,2)
  integer, intent(in)       :: element_type
  real(kind=dp)             :: jacobian(2,2)

  select case(element_type)
     case(0)
        jacobian = jacobian_spheroid(xi, eta, nodes)
     case(1)
        jacobian = jacobian_subpar(xi, eta, nodes)
     case(2)
        jacobian = jacobian_semino(xi, eta, nodes)
     case(3)
        jacobian = jacobian_semiso(xi, eta, nodes)
     case default
        write(6,*) 'ERROR: unknown element type: ', element_type
        stop
  end select

end function jacobian
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function inv_jacobian(xi, eta, nodes, element_type)

  real(kind=dp), intent(in) :: xi, eta, nodes(4,2)
  integer, intent(in)       :: element_type
  real(kind=dp)             :: inv_jacobian(2,2)

  select case(element_type)
     case(0)
        inv_jacobian = inv_jacobian_spheroid(xi, eta, nodes)
     case(1)
        inv_jacobian = inv_jacobian_subpar(xi, eta, nodes)
     case(2)
        inv_jacobian = inv_jacobian_semino(xi, eta, nodes)
     case(3)
        inv_jacobian = inv_jacobian_semiso(xi, eta, nodes)
     case default
        write(6,*) 'ERROR: unknown element type: ', element_type
        stop
  end select

end function inv_jacobian
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function jacobian_det(xi, eta, nodes, element_type, j)

  real(kind=dp), intent(in) :: xi, eta, nodes(4,2)
  integer, intent(in)       :: element_type
  real(kind=dp), intent(out), optional :: j(2,2)
  real(kind=dp)             :: jacobian_det
  real(kind=dp)             :: jacobian_buff(2,2)

  jacobian_buff = jacobian(xi, eta, nodes, element_type)

  jacobian_det = jacobian_buff(1,1) * jacobian_buff(2,2) &
               - jacobian_buff(2,1) * jacobian_buff(1,2)

  if (present(j)) j = jacobian_buff

end function jacobian_det
!-----------------------------------------------------------------------------------------


!!!!!!! SEMI SPHEROIDAL MAPPING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-----------------------------------------------------------------------------------------
pure function inv_mapping_semino(s, z, nodes)
!< This routines computes the coordinates in the reference coordinates xi, eta
!! this is nonlinear, but as the forward mapping is smooth, the gradient search
!! converges very quickly

  real(kind=dp), intent(in) :: s, z, nodes(4,2)
  real(kind=dp)             :: inv_mapping_semino(2), inv_jacobian(2,2)
  real(kind=dp)             :: xi, eta, sz(2), ds, dz
  integer                   :: i
  integer, parameter        :: numiter = 10

  ! starting value (center of the element)
  xi  = 0
  eta = 0

  do i=1, numiter
     sz = mapping_semino(xi, eta, nodes)

     ! distance in physical domain
     ds = s - sz(1)
     dz = z - sz(2)

     ! check convergence
     if ((ds**2 + dz**2) / (s**2 + z**2) < 1e-7**2) then
        exit
     endif

     inv_jacobian = inv_jacobian_semino(xi, eta, nodes)
     !                | dxi  / ds  dxi  / dz |
     ! inv_jacobian = |                      |
     !                | deta / ds  deta / dz |

     ! update
     xi  =  xi + inv_jacobian(1,1) * ds + inv_jacobian(1,2) * dz
     eta = eta + inv_jacobian(2,1) * ds + inv_jacobian(2,2) * dz
  enddo

  inv_mapping_semino = [xi, eta]

end function inv_mapping_semino
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function inv_mapping_semiso(s, z, nodes)
!< This routines computes the coordinates in the reference coordinates xi, eta
!! this is nonlinear, but as the forward mapping is smooth, the gradient search
!! converges very quickly

  real(kind=dp), intent(in) :: s, z, nodes(4,2)
  real(kind=dp)             :: inv_mapping_semiso(2), inv_jacobian(2,2)
  real(kind=dp)             :: xi, eta, sz(2), ds, dz
  integer                   :: i
  integer, parameter        :: numiter = 10

  ! starting value (center of the element)
  xi  = 0
  eta = 0

  do i=1, numiter
     sz = mapping_semiso(xi, eta, nodes)

     ! distance in physical domain
     ds = s - sz(1)
     dz = z - sz(2)

     ! check convergence
     if ((ds**2 + dz**2) / (s**2 + z**2) < 1e-7**2) then
        exit
     endif

     inv_jacobian = inv_jacobian_semiso(xi, eta, nodes)
     !                | dxi  / ds  dxi  / dz |
     ! inv_jacobian = |                      |
     !                | deta / ds  deta / dz |

     ! update
     xi  =  xi + inv_jacobian(1,1) * ds + inv_jacobian(1,2) * dz
     eta = eta + inv_jacobian(2,1) * ds + inv_jacobian(2,2) * dz
  enddo

  inv_mapping_semiso = [xi, eta]

end function inv_mapping_semiso
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function mapping_semino(xi, eta, nodes)
! We are working in polar coordinates here: theta is the latitude.
! semino: linear at bottom, curved at top

  real(kind=dp), intent(in) :: xi, eta
  real(kind=dp), intent(in) :: nodes(4,2)
  real(kind=dp)             :: mapping_semino(2)

  real(kind=dp)             :: a_top, b_top
  real(kind=dp)             :: thetabartop, dthetatop
  real(kind=dp)             :: s_bot, z_bot, s_top, z_top

  call compute_parameters_semino(nodes, a_top, b_top, thetabartop, dthetatop)
  call compute_sz_xi_sline_no(s_bot, z_bot, xi, nodes)
  call compute_sz_xi(s_top, z_top, xi, a_top, b_top, thetabartop, dthetatop)

  mapping_semino(1) = (s_bot + s_top) / 2 + eta * (s_top - s_bot) / 2
  mapping_semino(2) = (z_bot + z_top) / 2 + eta * (z_top - z_bot) / 2

end function mapping_semino
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function mapping_semiso(xi, eta, nodes)
! We are working in polar coordinates here: theta is the latitude.
! semiso: linear at top, curved at bottom

  real(kind=dp), intent(in) :: xi, eta
  real(kind=dp), intent(in) :: nodes(4,2)
  real(kind=dp)             :: mapping_semiso(2)

  real(kind=dp)             :: a_bot, b_bot
  real(kind=dp)             :: thetabarbot, dthetabot
  real(kind=dp)             :: s_bot, z_bot, s_top, z_top

  call compute_parameters_semiso(nodes, a_bot, b_bot, thetabarbot, dthetabot)
  call compute_sz_xi_sline_so(s_top, z_top, xi, nodes)
  call compute_sz_xi(s_bot, z_bot, xi, a_bot, b_bot, thetabarbot, dthetabot)

  mapping_semiso(1) = (s_bot + s_top) / 2 + eta * (s_top - s_bot) / 2
  mapping_semiso(2) = (z_bot + z_top) / 2 + eta * (z_top - z_bot) / 2

end function mapping_semiso
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function jacobian_semino(xi, eta, nodes)
! semino: linear at bottom, curved at top

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: jacobian_semino(2,2)

  real(kind=dp) :: a_top, b_top
  real(kind=dp) :: thetabar_top, dtheta_top
  real(kind=dp) :: s_bot, z_bot, s_top, z_top
  real(kind=dp) :: ds_botdxi, dz_botdxi
  real(kind=dp) :: ds_topdxi, dz_topdxi

  call compute_parameters_semino(nodes, a_top, b_top, thetabar_top, dtheta_top)

  call compute_sz_xi_sline_no(s_bot, z_bot, xi, nodes)
  call compute_sz_xi(s_top, z_top, xi, a_top, b_top, thetabar_top, dtheta_top)

  call compute_dsdxi_dzdxi_sline_no(ds_botdxi, dz_botdxi, nodes)
  call compute_dsdxi_dzdxi(ds_topdxi, dz_topdxi, xi, a_top, b_top, &
                           thetabar_top, dtheta_top)

  jacobian_semino(1,1) = (ds_botdxi + ds_topdxi) / 2 + eta * (ds_topdxi - ds_botdxi) / 2
  jacobian_semino(1,2) = (s_top - s_bot) / 2

  jacobian_semino(2,1) = (dz_botdxi + dz_topdxi) / 2 + eta * (dz_topdxi - dz_botdxi) / 2
  jacobian_semino(2,2) = (z_top - z_bot) / 2

end function jacobian_semino
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function jacobian_semiso(xi, eta, nodes)
! semino: linear at to,p curved at bottom

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: jacobian_semiso(2,2)

  real(kind=dp) :: a_bot, b_bot
  real(kind=dp) :: thetabar_bot, dtheta_bot
  real(kind=dp) :: s_bot, z_bot, s_top, z_top
  real(kind=dp) :: ds_topdxi, dz_topdxi
  real(kind=dp) :: ds_botdxi, dz_botdxi

  call compute_parameters_semiso(nodes, a_bot, b_bot, thetabar_bot, dtheta_bot)
  call compute_sz_xi_sline_so(s_top, z_top, xi, nodes)
  call compute_sz_xi(s_bot, z_bot, xi, a_bot, b_bot, thetabar_bot, dtheta_bot)

  call compute_dsdxi_dzdxi(ds_botdxi, dz_botdxi, xi, a_bot, b_bot, &
                           thetabar_bot, dtheta_bot)
  call compute_dsdxi_dzdxi_sline_so(ds_topdxi, dz_topdxi, nodes)

  jacobian_semiso(1,1) = (ds_botdxi + ds_topdxi) / 2 + eta * (ds_topdxi - ds_botdxi) / 2
  jacobian_semiso(1,2) = (s_top - s_bot) / 2

  jacobian_semiso(2,1) = (dz_botdxi + dz_topdxi) / 2 + eta * (dz_topdxi - dz_botdxi) / 2
  jacobian_semiso(2,2) = (z_top - z_bot) / 2

end function jacobian_semiso
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function inv_jacobian_semino(xi, eta, nodes)

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: inv_jacobian_semino(2,2)
  real(kind=dp)              :: jacobian(2,2), jacobian_det

  jacobian = jacobian_semino(xi, eta, nodes)
  jacobian_det = jacobian(1,1) * jacobian(2,2) - jacobian(2,1) * jacobian(1,2)

  inv_jacobian_semino(1,1) =   jacobian(2,2) / jacobian_det
  inv_jacobian_semino(2,1) = - jacobian(2,1) / jacobian_det
  inv_jacobian_semino(1,2) = - jacobian(1,2) / jacobian_det
  inv_jacobian_semino(2,2) =   jacobian(1,1) / jacobian_det

end function inv_jacobian_semino
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function inv_jacobian_semiso(xi, eta, nodes)

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: inv_jacobian_semiso(2,2)
  real(kind=dp)              :: jacobian(2,2), jacobian_det

  jacobian = jacobian_semiso(xi, eta, nodes)
  jacobian_det = jacobian(1,1) * jacobian(2,2) - jacobian(2,1) * jacobian(1,2)

  inv_jacobian_semiso(1,1) =   jacobian(2,2) / jacobian_det
  inv_jacobian_semiso(2,1) = - jacobian(2,1) / jacobian_det
  inv_jacobian_semiso(1,2) = - jacobian(1,2) / jacobian_det
  inv_jacobian_semiso(2,2) =   jacobian(1,1) / jacobian_det

end function inv_jacobian_semiso
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_dsdxi_dzdxi(dsdxi, dzdxi, xi, a, b, thetabar, dtheta)

  real(kind=dp), intent(out) :: dsdxi, dzdxi
  real(kind=dp), intent(in)  :: xi, a, b, thetabar, dtheta

  dsdxi = -a * dtheta * dsin(thetabar + xi * dtheta / 2) / 2
  dzdxi =  b * dtheta * dcos(thetabar + xi * dtheta / 2) / 2

end subroutine compute_dsdxi_dzdxi
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_dsdxi_dzdxi_sline_no(dsdxi, dzdxi, nodes)

  real(kind=dp), intent(out) :: dsdxi, dzdxi
  real(kind=dp), intent(in)  :: nodes(4,2)

  dsdxi = (nodes(2,1) - nodes(1,1)) / 2
  dzdxi = (nodes(2,2) - nodes(1,2)) / 2

end subroutine compute_dsdxi_dzdxi_sline_no
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_dsdxi_dzdxi_sline_so(dsdxi, dzdxi, nodes)

  real(kind=dp), intent(out) :: dsdxi, dzdxi
  real(kind=dp), intent(in)  :: nodes(4,2)

  dsdxi = (nodes(3,1) - nodes(4,1)) / 2
  dzdxi = (nodes(3,2) - nodes(4,2)) / 2

end subroutine compute_dsdxi_dzdxi_sline_so
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_sz_xi_sline_no(s, z, xi, nodes)

  real(kind=dp), intent(out) :: s,z
  real(kind=dp), intent(in)  :: xi, nodes(4,2)

  s = ((1 + xi) * nodes(2,1) + (1 - xi) * nodes(1,1)) / 2
  z = ((1 + xi) * nodes(2,2) + (1 - xi) * nodes(1,2)) / 2

end subroutine compute_sz_xi_sline_no
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_sz_xi_sline_so(s, z, xi, nodes)

  real(kind=dp), intent(out) :: s, z
  real(kind=dp), intent(in)  :: xi, nodes(4,2)

  s = ((1 + xi) * nodes(3,1) + (1 - xi) * nodes(4,1)) / 2
  z = ((1 + xi) * nodes(3,2) + (1 - xi) * nodes(4,2)) / 2

end subroutine compute_sz_xi_sline_so
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_sz_xi(s, z, xi, a, b, thetabar, dtheta)

  real(kind=dp), intent(out) :: s,z
  real(kind=dp), intent(in)  :: xi, a, b, thetabar, dtheta

  s = a * dcos(thetabar + xi * dtheta / 2)
  z = b * dsin(thetabar + xi * dtheta / 2)

end subroutine compute_sz_xi
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_parameters_semino(nodes, atop, btop, thetabartop, dthetatop)

  real(kind=dp), intent(in)  :: nodes(4,2)
  real(kind=dp), intent(out) :: atop, btop
  real(kind=dp), intent(out) :: thetabartop, dthetatop
  real(kind=dp)              :: theta3, theta4

  call compute_ab(atop, btop, nodes(4,1), nodes(4,2), nodes(3,1), nodes(3,2))
  theta3 = compute_theta(nodes(3,1), nodes(3,2), atop, btop)
  theta4 = compute_theta(nodes(4,1), nodes(4,2), atop, btop)

  thetabartop = (theta4 + theta3) / 2
  dthetatop = theta3 - theta4

end subroutine compute_parameters_semino
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_parameters_semiso(nodes, abot, bbot, thetabarbot, dthetabot)

  real(kind=dp), intent(in)  :: nodes(4,2)
  real(kind=dp), intent(out) :: abot, bbot
  real(kind=dp), intent(out) :: thetabarbot, dthetabot
  real(kind=dp)              :: theta1, theta2

  call compute_ab(abot, bbot, nodes(1,1), nodes(1,2), nodes(2,1), nodes(2,2))
  theta2 = compute_theta(nodes(2,1), nodes(2,2), abot, bbot)
  theta1 = compute_theta(nodes(1,1), nodes(1,2), abot, bbot)

  thetabarbot = (theta1 + theta2) / 2
  dthetabot = theta2 - theta1

end subroutine compute_parameters_semiso
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_ab(a, b, s1, z1, s2, z2)

  real(kind=dp), intent(out) :: a, b
  real(kind=dp), intent(in)  :: s1, z1, s2, z2

  a = dsqrt(dabs((s2**2 * z1**2 - z2**2 * s1**2) / (z1**2 - z2**2)))
  b = dsqrt(dabs((z1**2 * s2**2 - z2**2 * s1**2) / (s2**2 - s1**2)))

end subroutine compute_ab
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function compute_theta(s, z, a, b)
!	This routine returns the latitude theta, given s and z.

  real(kind=dp), intent(in)  :: s, z, a, b
  real(kind=dp)              :: compute_theta

  if (s >= 0d0) then
     compute_theta = datan2(z*a, s*b)
  else
     if (z > 0) compute_theta =  pi / 2
     if (z < 0) compute_theta = -pi / 2
  end if

end function compute_theta
!-----------------------------------------------------------------------------------------

!!!!!!! SPHEROIDAL MAPPING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-----------------------------------------------------------------------------------------
pure function inv_mapping_spheroid(s, z, nodes)
!< This routines computes the coordinates in the reference coordinates xi, eta
!! this is nonlinear, but as the forward mapping is smooth, the gradient search
!! converges very quickly

  real(kind=dp), intent(in) :: s, z, nodes(4,2)
  real(kind=dp)             :: inv_mapping_spheroid(2), inv_jacobian(2,2)
  real(kind=dp)             :: xi, eta, sz(2), ds, dz
  integer                   :: i
  integer, parameter        :: numiter = 10

  ! starting value (center of the element)
  xi  = 0
  eta = 0

  do i=1, numiter
     sz = mapping_spheroid(xi, eta, nodes)

     ! distance in physical domain
     ds = s - sz(1)
     dz = z - sz(2)

     ! check convergence
     if ((ds**2 + dz**2) / (s**2 + z**2) < 1e-7**2) then
        exit
     endif

     inv_jacobian = inv_jacobian_spheroid(xi, eta, nodes)
     !                | dxi  / ds  dxi  / dz |
     ! inv_jacobian = |                      |
     !                | deta / ds  deta / dz |

     ! update
     xi  =  xi + inv_jacobian(1,1) * ds + inv_jacobian(1,2) * dz
     eta = eta + inv_jacobian(2,1) * ds + inv_jacobian(2,2) * dz
  enddo

  inv_mapping_spheroid = [xi, eta]

end function inv_mapping_spheroid
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function mapping_spheroid(xi, eta, nodes)
!< The mapping for type (A) in paper 2.

  real(kind=dp), intent(in)   :: xi, eta
  real(kind=dp), intent(in)   :: nodes(4,2)
  real(kind=dp)               :: mapping_spheroid(2)
  real(kind=dp)               :: theta(4), r(4)

  call compute_theta_r(theta, r, nodes)

  mapping_spheroid(1) &
    = (1 + eta) * r(4) / 2 * dsin(((1 - xi) * theta(4) + (1 + xi) * theta(3)) / 2) &
    + (1 - eta) * r(1) / 2 * dsin(((1 - xi) * theta(1) + (1 + xi) * theta(2)) / 2)

  mapping_spheroid(2) &
    = (1 + eta) * r(4) / 2 * dcos(((1 - xi) * theta(4) + (1 + xi) * theta(3)) / 2) &
    + (1 - eta) * r(1) / 2 * dcos(((1 - xi) * theta(1) + (1 + xi) * theta(2)) / 2)

end function mapping_spheroid
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function jacobian_spheroid(xi, eta, nodes)

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: jacobian_spheroid(2,2)
  real(kind=dp)              :: theta(4), r(4)

  call compute_theta_r(theta, r, nodes)

  jacobian_spheroid(1,1) &
    = (1 + eta) * r(4) * (theta(3) - theta(4)) / 4  &
         * dcos(((1 - xi) * theta(4) + (1 + xi) * theta(3)) / 2) &
    + (1 - eta) * r(1) * (theta(2) - theta(1)) / 4  &
         * dcos(((1 - xi) * theta(1) + (1 + xi) * theta(2)) / 2)

  jacobian_spheroid(1,2) &
    = 0.5d0 * ( r(4) * dsin(((1 - xi) * theta(4) + (1 + xi) * theta(3)) / 2) &
              - r(1) * dsin(((1 - xi) * theta(1) + (1 + xi) * theta(2)) / 2))

  jacobian_spheroid(2,1) &
    = - (1 + eta) * r(4) * (theta(3) - theta(4)) / 4 &
       * dsin(((1 - xi) * theta(4) + (1 + xi) * theta(3)) / 2) &
      - (1 - eta) * r(1) * (theta(2) - theta(1)) / 4  &
       * dsin(((1 - xi) * theta(1) + (1 + xi) * theta(2)) / 2)

  jacobian_spheroid(2,2) &
    = 0.5d0 * ( r(4) * dcos(((1 - xi) * theta(4) + (1 + xi) * theta(3)) / 2) &
              - r(1) * dcos(((1 - xi) * theta(1) + (1 + xi) * theta(2)) / 2))

end function jacobian_spheroid
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function inv_jacobian_spheroid(xi, eta, nodes)

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: inv_jacobian_spheroid(2,2)
  real(kind=dp)              :: jacobian(2,2), jacobian_det

  jacobian = jacobian_spheroid(xi, eta, nodes)
  jacobian_det = jacobian(1,1) * jacobian(2,2) - jacobian(2,1) * jacobian(1,2)

  inv_jacobian_spheroid(1,1) =   jacobian(2,2) / jacobian_det
  inv_jacobian_spheroid(2,1) = - jacobian(2,1) / jacobian_det
  inv_jacobian_spheroid(1,2) = - jacobian(1,2) / jacobian_det
  inv_jacobian_spheroid(2,2) =   jacobian(1,1) / jacobian_det

end function inv_jacobian_spheroid
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure subroutine compute_theta_r(theta, r, nodes)

  real(kind=dp), dimension(4,2), intent(in) :: nodes
  real(kind=dp), dimension(4), intent(out)  :: theta, r
  integer                                   :: i

  do i=1,4
    r(i) = sqrt(nodes(i,1)**2 + nodes(i,2)**2)

    if (r(i) /= 0) then
       theta(i) = dacos(nodes(i,2) / r(i))
    else
       theta(i) = 0
    end if
  enddo

end subroutine compute_theta_r
!-----------------------------------------------------------------------------------------


!!!!!!! SUBPAR MAPPING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-----------------------------------------------------------------------------------------
pure function inv_mapping_subpar(s, z, nodes)
!< This routines computes the coordinates in the reference coordinates xi, eta
!! this is nonlinear, but as the forward mapping is smooth, the gradient search
!! converges very quickly

  real(kind=dp), intent(in) :: s, z, nodes(4,2)
  real(kind=dp)             :: inv_mapping_subpar(2), inv_jacobian(2,2)
  real(kind=dp)             :: xi, eta, sz(2), ds, dz
  integer                   :: i
  integer, parameter        :: numiter = 10

  ! starting value (center of the element)
  xi  = 0
  eta = 0

  do i=1, numiter
     sz = mapping_subpar(xi, eta, nodes)

     ! distance in physical domain
     ds = s - sz(1)
     dz = z - sz(2)

     ! check convergence
     if ((ds**2 + dz**2) / (s**2 + z**2) < 1e-7**2) then
        exit
     endif

     inv_jacobian = inv_jacobian_subpar(xi, eta, nodes)
     !                | dxi  / ds  dxi  / dz |
     ! inv_jacobian = |                      |
     !                | deta / ds  deta / dz |

     ! update
     xi  =  xi + inv_jacobian(1,1) * ds + inv_jacobian(1,2) * dz
     eta = eta + inv_jacobian(2,1) * ds + inv_jacobian(2,2) * dz
  enddo

  inv_mapping_subpar = [xi, eta]

end function inv_mapping_subpar
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function mapping_subpar(xi, eta, nodes)
!< This routines computes the coordinates of
!! any point in the reference domain in the physical domain.
!
! Topology is defined as follows
!
! 4 - - - - - - - 3
! |       ^       |
! |   eta |       |
! |       |       |
! |        --->   |
! |        xi     |
! |               |
! |               |
! 1 - - - - - - - 2

  real(kind=dp), intent(in) :: xi, eta, nodes(4,2)
  real(kind=dp)             :: mapping_subpar(2)
  integer                   :: inode
  real(kind=dp)             :: shp(4)

  ! Compute the shape functions

  shp = shp4(xi, eta)

  mapping_subpar = 0

  do inode = 1, 4
     mapping_subpar(:) = mapping_subpar(:) + shp(inode) * nodes(inode, :)
  end do

end function mapping_subpar
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function jacobian_subpar(xi, eta, nodes)

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: jacobian_subpar(2,2)
  integer                    :: inode
  real(kind=dp)              :: shpder(4,2)

  ! Compute the  derivatives of the shape functions
  shpder = shp4der(xi, eta)

  jacobian_subpar = 0

  do inode = 1, 4
     jacobian_subpar(1,1) = jacobian_subpar(1,1) &
                              + nodes(inode,1) * shpder(inode,1)
     jacobian_subpar(1,2) = jacobian_subpar(1,2) &
                              + nodes(inode,1) * shpder(inode,2)
     jacobian_subpar(2,1) = jacobian_subpar(2,1) &
                              + nodes(inode,2) * shpder(inode,1)
     jacobian_subpar(2,2) = jacobian_subpar(2,2) &
                              + nodes(inode,2) * shpder(inode,2)
  end do

end function jacobian_subpar
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function inv_jacobian_subpar(xi, eta, nodes)

  real(kind=dp), intent(in)  :: xi, eta, nodes(4,2)
  real(kind=dp)              :: inv_jacobian_subpar(2,2)
  real(kind=dp)              :: jacobian(2,2), jacobian_det

  jacobian = jacobian_subpar(xi, eta, nodes)
  jacobian_det = jacobian(1,1) * jacobian(2,2) - jacobian(2,1) * jacobian(1,2)

  inv_jacobian_subpar(1,1) =   jacobian(2,2) / jacobian_det
  inv_jacobian_subpar(2,1) = - jacobian(2,1) / jacobian_det
  inv_jacobian_subpar(1,2) = - jacobian(1,2) / jacobian_det
  inv_jacobian_subpar(2,2) =   jacobian(1,1) / jacobian_det

end function inv_jacobian_subpar
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function shp4(xi, eta)
!< This routine computes and returns the linear
!! shape functions axixiociated with a 4-nodes
!! element for a given point of coordinates (xi,eta).
!
  real(kind=dp), intent(in)  :: xi, eta
  real(kind=dp)              :: shp4(4)
  real(kind=dp)              :: xip, xim, etap, etam

  xip  = 1 +  xi
  xim  = 1 -  xi
  etap = 1 + eta
  etam = 1 - eta

  shp4(1) = xim * etam / 4
  shp4(2) = xip * etam / 4
  shp4(3) = xip * etap / 4
  shp4(4) = xim * etap / 4

end function shp4
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
pure function shp4der(xi, eta)
!< This routine computes and returns the derivatives
!! of the shape functions axixiociated with a 4-nodes
!! element for a given point of coordinates (xi,eta).

  real(kind=dp), intent(in)  :: xi, eta
  real(kind=dp)              :: shp4der(4,2)
  real(kind=dp)              :: xip, xim, etap, etam

  xip  = 1 +  xi
  xim  = 1 -  xi
  etap = 1 + eta
  etam = 1 - eta

  ! derivatives with respect to xi
  shp4der(1,1) = - etam / 4
  shp4der(2,1) =   etam / 4
  shp4der(3,1) =   etap / 4
  shp4der(4,1) = - etap / 4

  ! derivatives with respect to eta
  shp4der(1,2) = - xim / 4
  shp4der(2,2) = - xip / 4
  shp4der(3,2) =   xip / 4
  shp4der(4,2) =   xim / 4

end function shp4der
!-----------------------------------------------------------------------------------------


end module
!=========================================================================================
