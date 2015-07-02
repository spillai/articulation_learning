find $1 -type f \( -iname "lcm*" ! -iname "*jlp" ! -iname "*avi"  ! -iname "*.h5" \) -printf "python gpft_tracking.py -l %p\n" | bash
