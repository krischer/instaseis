!=========================================================================================
! copyright:
!     Martin van Driel (Martin@vanDriel.de), 2020
!     Lion Krischer (lion.krischer@gmail.com), 2020
! license:
!     GNU Lesser General Public License, Version 3 [non-commercial/academic use]
!     (http://www.gnu.org/copyleft/lgpl.html)

module global_parameters

  implicit none
  public
  integer, parameter         :: sp = selected_real_kind(6, 37)
  integer, parameter         :: dp = selected_real_kind(15, 307)

  real(kind=dp), parameter   :: pi = 3.1415926535898D0
end module
!=========================================================================================
