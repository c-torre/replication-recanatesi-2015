;; Set of GNU Guix channels to use to reproduce this computational
;; experiment.  Run:
;;
;;   guix time-machine -C channels.scm -- shell -m manifest.scm
;;
;; to enter the environment built from this very Guix revision.

(list (channel
        (name 'guix)
        (url "https://git.savannah.gnu.org/git/guix.git")
        (branch "master")
        (commit
          "dc90c0807d0a46cdd4b0a2c2b3f9becca9f97285")
        (introduction
          (make-channel-introduction
            "9edb3f66fd807b096b48283debdcddccfea34bad"
            (openpgp-fingerprint
              "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA")))))
