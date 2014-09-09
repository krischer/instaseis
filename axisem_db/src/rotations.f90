!=========================================================================================
module rotations

    use global_parameters, only: sp, dp, pi
    use iso_c_binding, only: c_double, c_int

    implicit none
    private

    public                                :: rotate_straintensor        ! only public for tests
    public                                :: rotate_straintensor_voigt  ! only public for tests
    public                                :: rotate_frame_rd            ! only public for tests
    public                                :: rotate_symm_tensor_voigt_src_to_xyz
    public                                :: rotate_symm_tensor_voigt_xyz_to_src
    public                                :: rotate_symm_tensor_voigt_xyz_src_to_xyz_earth
    public                                :: rotate_symm_tensor_voigt_xyz_earth_to_xyz_src

    public                                :: azim_factor
    public                                :: azim_factor_bw

    interface rotate_symm_tensor_voigt_src_to_xyz
      module procedure  :: rotate_symm_tensor_voigt_src_to_xyz_1d
      module procedure  :: rotate_symm_tensor_voigt_src_to_xyz_2d
    end interface

    interface rotate_symm_tensor_voigt_xyz_to_src
      module procedure  :: rotate_symm_tensor_voigt_xyz_to_src_1d
      module procedure  :: rotate_symm_tensor_voigt_xyz_to_src_2d
    end interface

    interface rotate_symm_tensor_voigt_xyz_src_to_xyz_earth
      module procedure  :: rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d
      module procedure  :: rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_2d
    end interface

    interface rotate_symm_tensor_voigt_xyz_earth_to_xyz_src
      module procedure  :: rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d
      module procedure  :: rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_2d
    end interface

contains


!--C Wrappers-------------------------------------------------------------------------------
!subroutine rotate_straintensor_wrapped(tensor_vector, phi, mij, isim, tensor_return) &
!    bind(c, name="rotate_straintensor")
!    real(kind=dp), intent(in)    :: tensor_vector(:,:)
!    real(kind=dp), intent(in)    :: phi, mij(6)
!    integer      , intent(in)    :: isim
!
!    real(kind=dp), intent(out)   :: tensor_return(:,:)
!
!    tensor_return =  rotate_straintensor(tensor_vector, phi, mij, isim)
!end subroutine rotate_straintensor_wrapped


subroutine rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d_wrapped(tensor_voigt, phi, theta, &
                                                                    tensor_return) &
    bind(c, name="rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d")

    real(c_double), intent(in)        :: tensor_voigt(6)
    real(c_double), intent(in), value :: phi, theta
    real(c_double)                    :: tensor_return(6)

    tensor_return = rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(tensor_voigt, phi, theta)
end subroutine


subroutine rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d_wrapped(tensor_voigt, phi, theta, &
                                                                    tensor_return) &
    bind(c, name="rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d")

    real(c_double), intent(in)        :: tensor_voigt(6)
    real(c_double), intent(in), value :: phi, theta
    real(c_double)                    :: tensor_return(6)

    tensor_return = rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(tensor_voigt, phi, theta)
end subroutine


subroutine rotate_symm_tensor_voigt_xyz_to_src_1d_wrapped(tensor_voigt, phi, tensor_return) &
    bind(c, name="rotate_symm_tensor_voigt_xyz_to_src_1d")

    real(c_double), intent(in)        :: tensor_voigt(6)
    real(c_double), intent(in), value :: phi
    real(c_double)                    :: tensor_return(6)

    tensor_return = rotate_symm_tensor_voigt_xyz_to_src_1d(tensor_voigt, phi)
end subroutine


!-----------------------------------------------------------------------------------------
function azim_factor(phi, mij, isim, ikind)

    real(kind=dp), intent(in)    :: phi
    real(kind=dp), intent(in)    :: mij(6)  ! rr, tt, pp, rt, rp, tp
    integer, intent(in)          :: isim, ikind
    real(kind=dp)                :: azim_factor

    !@TODO: is isim a robust indicator for the souretype? what about a single
    !       simulation that is not Mzz? (MvD)
    select case(isim)
    case(1) ! Mzz
       if (ikind==1) then
           azim_factor = Mij(1)
       else
           azim_factor = 0
       end if

    case(2) ! Mxx+Myy
       if (ikind==1) then
           azim_factor = 0.5d0 * (Mij(2) + Mij(3))
       else
           azim_factor = 0
       end if

    case(3) ! dipole
       if (ikind==1) then
           azim_factor =   Mij(4) * dcos(phi) + Mij(5) * dsin(phi)
       else
           azim_factor = - Mij(4) * dsin(phi) + Mij(5) * dcos(phi)
       end if

    case(4) ! quadrupole
       if (ikind==1) then
           azim_factor =   0.5d0 * (Mij(2) - Mij(3)) * dcos(2.d0 * phi) &
                            + Mij(6) * dsin(2.d0 * phi)
       else
           azim_factor = - 0.5d0 * (Mij(2) - Mij(3)) * dsin(2.d0 * phi) &
                            + Mij(6) * dcos(2.d0 * phi)
       end if

    case default
       write(6,*) ': unknown number of simulations',isim
       call abort
    end select

end function
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function azim_factor_bw(phi, fi, isim, ikind) bind(c, name="azim_factor_bw")

    real(c_double), intent(in), value  :: phi
    real(c_double), intent(in)         :: fi(3) ! r, t, p
    integer(c_int), intent(in), value  :: isim, ikind
    real(c_double)                     :: azim_factor_bw

    !@TODO: is isim a robust indicator for the souretype? what about a single
    !       simulation that is not Mzz? (MvD)
    select case(isim)
    case(1) ! vertforce
       if (ikind==1) then
           azim_factor_bw = fi(1)
       else
           azim_factor_bw = 0
       end if

    case(2) ! horizontal force
       if (ikind==1) then
           azim_factor_bw =   fi(2) * dcos(phi) + fi(3) * dsin(phi)
       else
           azim_factor_bw = - fi(2) * dsin(phi) + fi(3) * dcos(phi)
       end if

    case default
       write(6,*) ': unknown number of simulations',isim
       call abort
    end select

end function
!-----------------------------------------------------------------------------------------


!-----------------------------------------------------------------------------------------
function rotate_straintensor(tensor_vector, phi, mij, isim) result(tensor_return)
    !! 'strain_dsus', 'strain_dsuz', 'strain_dpup', &
    !! 'strain_dsup', 'strain_dzup', 'straintrace']
    real(kind=dp), intent(in)    :: tensor_vector(:,:)
    real(kind=dp), intent(in)    :: phi, mij(6)
    integer      , intent(in)    :: isim

    real(kind=dp), allocatable   :: tensor_return(:,:)

    real(kind=dp), allocatable   :: tensor_matrix(:,:,:)
    real(kind=dp)                :: conv_mat(3,3)     !from s,z,phi to x,y,z
    real(kind=dp)                :: azim_factor_1, azim_factor_2
    integer                      :: idump

    !print *, 'Length of tensor_vector: ', size(tensor_vector,1), size(tensor_vector,2)
    if (size(tensor_vector,2).ne.6) then
        print *, 'ERROR in rotate_straintensor: size of second dimension of tensor_vector:'
        print *, 'should be: 6, is: ', size(tensor_vector, 2)
    end if

    allocate(tensor_return(size(tensor_vector, 1), 6))
    allocate(tensor_matrix(3, 3, size(tensor_vector, 1)))

    azim_factor_1 = azim_factor(phi, mij, isim, 1)
    azim_factor_2 = azim_factor(phi, mij, isim, 2)

    do idump = 1, size(tensor_vector, 1)
        tensor_matrix(1,1,idump) =  tensor_vector(idump,1) * azim_factor_1  !ss
        tensor_matrix(1,2,idump) =  tensor_vector(idump,2) * azim_factor_2  !sz
        tensor_matrix(1,3,idump) =  tensor_vector(idump,4) * azim_factor_1  !sp
        tensor_matrix(2,1,idump) =  tensor_matrix(1,2,idump)                !zs
        tensor_matrix(2,2,idump) = (tensor_vector(idump,6) - &
                                    tensor_vector(idump,1) - &
                                    tensor_vector(idump,3)) * azim_factor_1 !zz
        tensor_matrix(2,3,idump) =  tensor_vector(idump,5) * azim_factor_2  !zp
        tensor_matrix(3,1,idump) =  tensor_matrix(1,3,idump)                !ps
        tensor_matrix(3,2,idump) =  tensor_matrix(2,3,idump)                !pz
        tensor_matrix(3,3,idump) =  tensor_vector(idump,3) * azim_factor_1  !pp
    end do


    ! Conversion to cartesian coordinates, from s,z,phi to x,y,z
    conv_mat(1,:) = [ dcos(phi), dsin(phi),       0.0d0]
    conv_mat(2,:) = [     0.0d0,     0.0d0,       1.0d0]
    conv_mat(3,:) = [-dsin(phi), dcos(phi),       0.0d0]

    do idump = 1, size(tensor_vector, 1)
        tensor_matrix(:,:,idump) = matmul(matmul(transpose(conv_mat),       &
                                                 tensor_matrix(:,:,idump)), &
                                          conv_mat)
        ! seriously? can this happen? just 20 lines earlier, this tensor is
        ! hardcoded symmetric (MvD)
        if (abs(tensor_matrix(1,3,idump)-tensor_matrix(3,1,idump)) /        &
            abs(tensor_matrix(1,3,idump))>1e-5) then
            print *, 'nonsymmetric strain components (1,3) at dump', idump
            print *, '(1,3),(3,1):',tensor_matrix(1,3,idump),tensor_matrix(3,1,idump)
        end if
    end do


    tensor_return(:,1) = tensor_matrix(1,1,:) !xx
    tensor_return(:,2) = tensor_matrix(2,2,:) !yy
    tensor_return(:,3) = tensor_matrix(3,3,:) !zz
    tensor_return(:,4) = tensor_matrix(1,2,:) !xy
    tensor_return(:,5) = tensor_matrix(1,3,:) !xz
    tensor_return(:,6) = tensor_matrix(2,3,:) !yz

    ! OK, now we have the tensor in a cartesian system where the z-axis is aligned with
    ! the source. Needs to be further rotated to cartesian system with z aligned
    ! to north pole.

end function rotate_straintensor
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_straintensor_voigt(tensor_vector, phi, mij, isim) result(tensor_return)
    real(kind=dp), intent(in)    :: tensor_vector(:,:) ! in voigt mapping
    real(kind=dp), intent(in)    :: phi, mij(6)
    integer      , intent(in)    :: isim

    real(kind=dp), allocatable   :: tensor_return(:,:)

    real(kind=dp)                :: azim_factor_1, azim_factor_2

    !print *, 'Length of tensor_vector: ', size(tensor_vector,1), size(tensor_vector,2)
    if (size(tensor_vector,2).ne.6) then
        print *, 'ERROR in rotate_straintensor: size of second dimension of tensor_vector:'
        print *, 'should be: 6, is: ', size(tensor_vector, 2)
    end if

    allocate(tensor_return(size(tensor_vector, 1), 6))

    azim_factor_1 = azim_factor(phi, mij, isim, 1)
    azim_factor_2 = azim_factor(phi, mij, isim, 2)

    tensor_return(:,1) = azim_factor_1 * tensor_vector(:,1)
    tensor_return(:,2) = azim_factor_1 * tensor_vector(:,2)
    tensor_return(:,3) = azim_factor_1 * tensor_vector(:,3)
    tensor_return(:,4) = azim_factor_2 * tensor_vector(:,4)
    tensor_return(:,5) = azim_factor_1 * tensor_vector(:,5)
    tensor_return(:,6) = azim_factor_2 * tensor_vector(:,6)

    tensor_return(:,:) = rotate_symm_tensor_voigt_src_to_xyz_2d(tensor_return(:,:), phi, &
                                                                size(tensor_vector, 1))

end function rotate_straintensor_voigt
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_src_to_xyz_1d(tensor_voigt, phi) result(tensor_return)
    ! rotates a tensor from AxiSEM s, phi, z system aligned with the source to a
    ! cartesian system x,y,z where z is aligned with the source and x with phi = 0
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    !
    !        a1 a5 a6
    !   A =  a5 a2 a4
    !        a6 a4 a3
    !   which is easy to remember using the circular Voigt mapping 4 -> 23, 5 -> 31, 6 -> 12
    !
    ! rotation matrix
    ! R = {{Cos[phi], Sin[phi], 0}, {-Sin[phi] , Cos[phi], 0}, {0, 0, 1}};
    !
    ! compute and ouput in voigt notation:
    ! Rt.A.R
    !
    real(kind=dp), intent(in)    :: tensor_voigt(6)
    real(kind=dp), intent(in)    :: phi
    real(kind=dp)                :: tensor_return(6)
    real(kind=dp)                :: sp, cp

    sp = dsin(phi)
    cp = dcos(phi)

    tensor_return(1) = tensor_voigt(1) * cp ** 2 &
                        + sp * (-2 * tensor_voigt(6) * cp + tensor_voigt(2) * sp)
    tensor_return(2) = tensor_voigt(2) * cp ** 2 &
                        + sp * (2 * tensor_voigt(6) * cp + tensor_voigt(1) * sp)
    tensor_return(3) = tensor_voigt(3)
    tensor_return(4) = tensor_voigt(4) * cp + tensor_voigt(5) * sp
    tensor_return(5) = tensor_voigt(5) * cp - tensor_voigt(4) * sp
    tensor_return(6) = (tensor_voigt(1) - tensor_voigt(2)) * cp * sp &
                        + tensor_voigt(6) * (cp ** 2 - sp ** 2)

end function rotate_symm_tensor_voigt_src_to_xyz_1d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_src_to_xyz_2d(tensor_voigt, phi, npoint) result(tensor_return)
    ! rotates a tensor from AxiSEM s, phi, z system aligned with the source to a
    ! cartesian system x,y,z where z is aligned with the source and x with phi = 0
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    !
    !        a1 a5 a6
    !   A =  a5 a2 a4
    !        a6 a4 a3
    !   which is easy to remember using the circular Voigt mapping 4 -> 23, 5 -> 31, 6 -> 12
    !
    ! rotation matrix
    ! R = {{Cos[phi], Sin[phi], 0}, {-Sin[phi] , Cos[phi], 0}, {0, 0, 1}};
    !
    ! compute and ouput in voigt notation:
    ! Rt.A.R
    !
    real(kind=dp), intent(in)    :: tensor_voigt(npoint,6)
    real(kind=dp), intent(in)    :: phi
    integer, intent(in)          :: npoint
    real(kind=dp)                :: tensor_return(npoint,6)
    real(kind=dp)                :: sp, cp

    sp = dsin(phi)
    cp = dcos(phi)

    tensor_return(:,1) = tensor_voigt(:,1) * cp ** 2 &
                        + sp * (-2 * tensor_voigt(:,6) * cp + tensor_voigt(:,2) * sp)
    tensor_return(:,2) = tensor_voigt(:,2) * cp ** 2 &
                        + sp * (2 * tensor_voigt(:,6) * cp + tensor_voigt(:,1) * sp)
    tensor_return(:,3) = tensor_voigt(:,3)
    tensor_return(:,4) = tensor_voigt(:,4) * cp + tensor_voigt(:,5) * sp
    tensor_return(:,5) = tensor_voigt(:,5) * cp - tensor_voigt(:,4) * sp
    tensor_return(:,6) = (tensor_voigt(:,1) - tensor_voigt(:,2)) * cp * sp &
                        + tensor_voigt(:,6) * (cp ** 2 - sp ** 2)

end function rotate_symm_tensor_voigt_src_to_xyz_2d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_xyz_to_src_1d(tensor_voigt, phi) result(tensor_return)
    ! rotates a tensor from a cartesian system x,y,z where z is aligned with the source
    ! and x with phi = 0 to the AxiSEM s, phi, z system aligned with the source on the
    ! s = 0 axis
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    ! rotation matrix
    ! R = {{Cos[phi], Sin[phi], 0}, {-Sin[phi] , Cos[phi], 0}, {0, 0, 1}};
    !
    ! compute and ouput in voigt notation:
    ! R.A.Rt
    !
    real(kind=dp), intent(in)    :: tensor_voigt(6)
    real(kind=dp), intent(in)    :: phi
    real(kind=dp)                :: tensor_return(6)
    real(kind=dp)                :: sp, cp

    sp = dsin(phi)
    cp = dcos(phi)

    tensor_return(1) = tensor_voigt(1) * cp ** 2 &
                            + sp * (2 * tensor_voigt(6) * cp + tensor_voigt(2) * sp)
    tensor_return(2) = tensor_voigt(2) * cp ** 2 &
                            + sp * (-2 * tensor_voigt(6) * cp + tensor_voigt(1) * sp)
    tensor_return(3) = tensor_voigt(3)
    tensor_return(4) = tensor_voigt(4) * cp - tensor_voigt(5) * sp
    tensor_return(5) = tensor_voigt(5) * cp + tensor_voigt(4) * sp
    tensor_return(6) = cp * (tensor_voigt(6) * cp + tensor_voigt(2) * sp) &
                            - sp * (tensor_voigt(1) * cp + tensor_voigt(6) * sp)


end function rotate_symm_tensor_voigt_xyz_to_src_1d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_xyz_to_src_2d(tensor_voigt, phi, npoint) result(tensor_return)
    ! rotates a tensor from a cartesian system x,y,z where z is aligned with the source
    ! and x with phi = 0 to the AxiSEM s, phi, z system aligned with the source on the
    ! s = 0 axis
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    ! rotation matrix
    ! R = {{Cos[phi], Sin[phi], 0}, {-Sin[phi] , Cos[phi], 0}, {0, 0, 1}};
    !
    ! compute and ouput in voigt notation:
    ! R.A.Rt
    !
    real(kind=dp), intent(in)    :: tensor_voigt(npoint,6)
    real(kind=dp), intent(in)    :: phi
    integer, intent(in)          :: npoint
    real(kind=dp)                :: tensor_return(npoint,6)
    real(kind=dp)                :: sp, cp

    sp = dsin(phi)
    cp = dcos(phi)

    tensor_return(:,1) = tensor_voigt(:,1) * cp ** 2 &
                            + sp * (2 * tensor_voigt(:,6) * cp + tensor_voigt(:,2) * sp)
    tensor_return(:,2) = tensor_voigt(:,2) * cp ** 2 &
                            + sp * (-2 * tensor_voigt(:,6) * cp + tensor_voigt(:,1) * sp)
    tensor_return(:,3) = tensor_voigt(:,3)
    tensor_return(:,4) = tensor_voigt(:,4) * cp - tensor_voigt(:,5) * sp
    tensor_return(:,5) = tensor_voigt(:,5) * cp + tensor_voigt(:,4) * sp
    tensor_return(:,6) = cp * (tensor_voigt(:,6) * cp + tensor_voigt(:,2) * sp) &
                            - sp * (tensor_voigt(:,1) * cp + tensor_voigt(:,6) * sp)

end function rotate_symm_tensor_voigt_xyz_to_src_2d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(tensor_voigt, phi, theta) result(tensor_return)
    ! rotates a tensor from a cartesian system xyz with z axis aligned with the source to a
    ! cartesian system x,y,z where z is aligned with the north pole
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    ! rotation matrix from TNM 2007 eq 14
    ! R = {{ct*cp, -sp, st*cp}, {ct*sp , cp, st*sp}, {-st, 0, ct}}
    !
    ! compute and ouput in voigt notation:
    ! Rt.A.R
    !
    real(kind=dp), intent(in)    :: tensor_voigt(6)
    real(kind=dp), intent(in)    :: phi, theta
    real(kind=dp)                :: tensor_return(6)
    real(kind=dp)                :: sp, cp, st, ct

    sp = dsin(phi)
    cp = dcos(phi)
    st = dsin(theta)
    ct = dcos(theta)

    tensor_return(1) = tensor_voigt(1) * cp ** 2 * ct ** 2 &
                        - 2 * tensor_voigt(6) * cp * ct * sp &
                        + tensor_voigt(2) * sp ** 2 &
                        + 2 * tensor_voigt(5) * cp ** 2 * ct * st &
                        - 2 * tensor_voigt(4) * cp * sp * st &
                        + tensor_voigt(3) * cp ** 2 * st ** 2

    tensor_return(2) = tensor_voigt(2) * cp ** 2 &
                        + sp * (2 * tensor_voigt(6) * cp * ct &
                                + tensor_voigt(1) * ct ** 2 * sp &
                                + st * (2 * tensor_voigt(4) * cp &
                                        + 2 * tensor_voigt(5) * ct * sp &
                                        + tensor_voigt(3) * sp * st))

    tensor_return(3) = tensor_voigt(3) * ct ** 2 &
                        + st * (-2 * tensor_voigt(5) * ct + tensor_voigt(1) * st)

    tensor_return(4) = ct * (tensor_voigt(4) * cp + tensor_voigt(5) * ct * sp &
                                + tensor_voigt(3) * sp * st) &
                        - st * (tensor_voigt(6) * cp + tensor_voigt(1) * ct * sp &
                                + tensor_voigt(5) * sp * st)

    tensor_return(5) = ct * (tensor_voigt(5) * cp * ct - tensor_voigt(4) * sp &
                                + tensor_voigt(3) * cp * st) &
                        - st * (tensor_voigt(1) * cp * ct - tensor_voigt(6) * sp &
                                + tensor_voigt(5) * cp * st)

    tensor_return(6) = cp * st * (tensor_voigt(4) * cp + tensor_voigt(5) * ct * sp &
                                    + tensor_voigt(3) * sp * st) &
                        - sp * (tensor_voigt(2) * cp + tensor_voigt(6) * ct * sp &
                                    + tensor_voigt(4) * sp * st) &
                        + cp * ct * (tensor_voigt(6) * cp + tensor_voigt(1) * ct * sp &
                                    + tensor_voigt(5) * sp * st)

end function rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_2d(tensor_voigt, phi, theta, npoint) result(tensor_return)
    ! rotates a tensor from a cartesian system xyz with z axis aligned with the source to a
    ! cartesian system x,y,z where z is aligned with the north pole
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    ! rotation matrix from TNM 2007 eq 14
    ! R = {{ct*cp, -sp, st*cp}, {ct*sp , cp, st*sp}, {-st, 0, ct}}
    !
    ! compute and ouput in voigt notation:
    ! Rt.A.R
    !
    real(kind=dp), intent(in)    :: tensor_voigt(npoint,6)
    real(kind=dp), intent(in)    :: phi, theta
    integer, intent(in)          :: npoint
    real(kind=dp)                :: tensor_return(npoint,6)
    real(kind=dp)                :: sp, cp, st, ct

    sp = dsin(phi)
    cp = dcos(phi)
    st = dsin(theta)
    ct = dcos(theta)

    tensor_return(:,1) = tensor_voigt(:,1) * cp ** 2 * ct ** 2 &
                        - 2 * tensor_voigt(:,6) * cp * ct * sp &
                        + tensor_voigt(:,2) * sp ** 2 &
                        + 2 * tensor_voigt(:,5) * cp ** 2 * ct * st &
                        - 2 * tensor_voigt(:,4) * cp * sp * st &
                        + tensor_voigt(:,3) * cp ** 2 * st ** 2

    tensor_return(:,2) = tensor_voigt(:,2) * cp ** 2 &
                        + sp * (2 * tensor_voigt(:,6) * cp * ct &
                                + tensor_voigt(:,1) * ct ** 2 * sp &
                                + st * (2 * tensor_voigt(:,4) * cp &
                                        + 2 * tensor_voigt(:,5) * ct * sp &
                                        + tensor_voigt(:,3) * sp * st))

    tensor_return(:,3) = tensor_voigt(:,3) * ct ** 2 &
                        + st * (-2 * tensor_voigt(:,5) * ct + tensor_voigt(:,1) * st)

    tensor_return(:,4) = ct * (tensor_voigt(:,4) * cp + tensor_voigt(:,5) * ct * sp &
                                + tensor_voigt(:,3) * sp * st) &
                        - st * (tensor_voigt(:,6) * cp + tensor_voigt(:,1) * ct * sp &
                                + tensor_voigt(:,5) * sp * st)

    tensor_return(:,5) = ct * (tensor_voigt(:,5) * cp * ct - tensor_voigt(:,4) * sp &
                                + tensor_voigt(:,3) * cp * st) &
                        - st * (tensor_voigt(:,1) * cp * ct - tensor_voigt(:,6) * sp &
                                + tensor_voigt(:,5) * cp * st)

    tensor_return(:,6) = cp * st * (tensor_voigt(:,4) * cp + tensor_voigt(:,5) * ct * sp &
                                    + tensor_voigt(:,3) * sp * st) &
                        - sp * (tensor_voigt(:,2) * cp + tensor_voigt(:,6) * ct * sp &
                                    + tensor_voigt(:,4) * sp * st) &
                        + cp * ct * (tensor_voigt(:,6) * cp + tensor_voigt(:,1) * ct * sp &
                                    + tensor_voigt(:,5) * sp * st)

end function rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_2d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(tensor_voigt, phi, theta) result(tensor_return)
    ! rotates a tensor from a cartesian system xyz with z axis aligned with the north pole to a
    ! cartesian system x,y,z where z is aligned with the source / receiver
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    ! rotation matrix from TNM 2007 eq 14
    ! R = {{ct*cp, -sp, st*cp}, {ct*sp , cp, st*sp}, {-st, 0, ct}}
    !
    ! compute and ouput in voigt notation:
    ! R.A.Rt
    !
    real(kind=dp), intent(in)    :: tensor_voigt(6)
    real(kind=dp), intent(in)    :: phi, theta
    real(kind=dp)                :: tensor_return(6)
    real(kind=dp)                :: sp, cp, st, ct

    sp = dsin(phi)
    cp = dcos(phi)
    st = dsin(theta)
    ct = dcos(theta)

    tensor_return(1) = tensor_voigt(1) * cp ** 2 * ct ** 2 &
                        + 2 * tensor_voigt(6) * cp * ct ** 2 * sp &
                        + tensor_voigt(2) * ct ** 2 * sp ** 2 &
                        - 2 * tensor_voigt(5) * cp * ct * st &
                        - 2 * tensor_voigt(4) * ct * sp * st &
                        + tensor_voigt(3) * st ** 2

    tensor_return(2) = tensor_voigt(2) * cp ** 2 &
                        + sp * (-2 * tensor_voigt(6) * cp + tensor_voigt(1) * sp)

    tensor_return(3) = tensor_voigt(3) * ct ** 2 &
                        + st * (2 * tensor_voigt(5) * cp * ct &
                                + 2 * tensor_voigt(4) * ct * sp &
                                + tensor_voigt(1) * cp ** 2 * st &
                                + 2 * tensor_voigt(6) * cp * sp * st &
                                + tensor_voigt(2) * sp ** 2 * st)

    tensor_return(4) = cp * (tensor_voigt(4) * ct + tensor_voigt(6) * cp * st &
                            + tensor_voigt(2) * sp * st) &
                        - sp * (tensor_voigt(5) * ct + tensor_voigt(1) * cp * st &
                                + tensor_voigt(6) * sp * st)

    tensor_return(5) = ct * sp * (tensor_voigt(4) * ct + tensor_voigt(6) * cp * st &
                                + tensor_voigt(2) * sp * st) &
                        - st * (tensor_voigt(3) * ct + tensor_voigt(5) * cp * st &
                                + tensor_voigt(4) * sp * st) &
                        + cp * ct * (tensor_voigt(5) * ct + tensor_voigt(1) * cp * st &
                                    + tensor_voigt(6) * sp * st)

    tensor_return(6) = cp * (tensor_voigt(6) * cp * ct + tensor_voigt(2) * ct * sp &
                            - tensor_voigt(4) * st) &
                        - sp * (tensor_voigt(1) * cp * ct + tensor_voigt(6) * ct * sp &
                                - tensor_voigt(5) * st)

end function rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
function rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_2d(tensor_voigt, phi, theta, npoint) result(tensor_return)
    ! rotates a tensor from a cartesian system xyz with z axis aligned with the north pole to a
    ! cartesian system x,y,z where z is aligned with the source / receiver
    !
    ! input symmetric tensor in voigt notation:
    ! A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    ! rotation matrix from TNM 2007 eq 14
    ! R = {{ct*cp, -sp, st*cp}, {ct*sp , cp, st*sp}, {-st, 0, ct}}
    !
    ! compute and ouput in voigt notation:
    ! R.A.Rt
    !
    real(kind=dp), intent(in)    :: tensor_voigt(npoint,6)
    real(kind=dp), intent(in)    :: phi, theta
    integer, intent(in)          :: npoint
    real(kind=dp)                :: tensor_return(npoint,6)
    real(kind=dp)                :: sp, cp, st, ct

    sp = dsin(phi)
    cp = dcos(phi)
    st = dsin(theta)
    ct = dcos(theta)

    tensor_return(:,1) = tensor_voigt(:,1) * cp ** 2 * ct ** 2 &
                        + 2 * tensor_voigt(:,6) * cp * ct ** 2 * sp &
                        + tensor_voigt(:,2) * ct ** 2 * sp ** 2 &
                        - 2 * tensor_voigt(:,5) * cp * ct * st &
                        - 2 * tensor_voigt(:,4) * ct * sp * st &
                        + tensor_voigt(:,3) * st ** 2

    tensor_return(:,2) = tensor_voigt(:,2) * cp ** 2 &
                        + sp * (-2 * tensor_voigt(:,6) * cp + tensor_voigt(:,1) * sp)

    tensor_return(:,3) = tensor_voigt(:,3) * ct ** 2 &
                        + st * (2 * tensor_voigt(:,5) * cp * ct &
                                + 2 * tensor_voigt(:,4) * ct * sp &
                                + tensor_voigt(:,1) * cp ** 2 * st &
                                + 2 * tensor_voigt(:,6) * cp * sp * st &
                                + tensor_voigt(:,2) * sp ** 2 * st)

    tensor_return(:,4) = cp * (tensor_voigt(:,4) * ct + tensor_voigt(:,6) * cp * st &
                            + tensor_voigt(:,2) * sp * st) &
                        - sp * (tensor_voigt(:,5) * ct + tensor_voigt(:,1) * cp * st &
                                + tensor_voigt(:,6) * sp * st)

    tensor_return(:,5) = ct * sp * (tensor_voigt(:,4) * ct + tensor_voigt(:,6) * cp * st &
                                + tensor_voigt(:,2) * sp * st) &
                        - st * (tensor_voigt(:,3) * ct + tensor_voigt(:,5) * cp * st &
                                + tensor_voigt(:,4) * sp * st) &
                        + cp * ct * (tensor_voigt(:,5) * ct + tensor_voigt(:,1) * cp * st &
                                    + tensor_voigt(:,6) * sp * st)

    tensor_return(:,6) = cp * (tensor_voigt(:,6) * cp * ct + tensor_voigt(:,2) * ct * sp &
                            - tensor_voigt(:,4) * st) &
                        - sp * (tensor_voigt(:,1) * cp * ct + tensor_voigt(:,6) * ct * sp &
                                - tensor_voigt(:,5) * st)

end function rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_2d
!-----------------------------------------------------------------------------------------

!-----------------------------------------------------------------------------------------
subroutine rotate_frame_rd(npts, srd, phird, zrd, rgd, phigr, thetagr)
    ! transforms coordinates from a global cartesian coordinate system (rgd) with
    ! northpole on the z axis to a cartesian system with source / receiver at phigr,
    ! thetagr on the z axis and then computes cylindrical coordinates s,phi,z
    !
    ! this is a passive transformation, i.e. change of coordinate system.

    implicit none
    integer, intent(in)                            :: npts
    !< Number of points to rotate

    real(kind=dp), dimension(3, npts), intent(in)  :: rgd
    !< Coordinates to rotate (in x, y, z)

    real(kind=dp), intent(in)                      :: phigr, thetagr
    !< Rotation angles phi and theta

    real(kind=dp), dimension(npts), intent(out)    :: srd, zrd, phird
    !< Rotated coordinates (in s, z, phi)

    real(kind=dp), dimension(npts)                 :: xp, yp, zp
    real(kind=dp), dimension(npts)                 :: xp_cp, yp_cp, zp_cp
    real(kind=dp)                                  :: phi_cp
    integer                                        :: ipt

    !first rotation (longitude)
    xp_cp =  rgd(1,:) * dcos(phigr) + rgd(2,:) * dsin(phigr)
    yp_cp = -rgd(1,:) * dsin(phigr) + rgd(2,:) * dcos(phigr)
    zp_cp =  rgd(3,:)

    !second rotation (colat)
    xp = xp_cp * dcos(thetagr) - zp_cp * dsin(thetagr)
    yp = yp_cp
    zp = xp_cp * dsin(thetagr) + zp_cp * dcos(thetagr)

    srd = dsqrt(xp*xp + yp*yp)
    zrd = zp
    do ipt = 1, npts
       phi_cp = datan2(yp(ipt), xp(ipt))
       if (phi_cp < 0.d0) then
          phird(ipt) = 2.d0 * pi + phi_cp
       else
          phird(ipt) = phi_cp
       endif
    enddo
end subroutine rotate_frame_rd
!-----------------------------------------------------------------------------------------


end module
!=========================================================================================
