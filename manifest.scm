;; GNU Guix manifest to set up the execution environment of this
;; computational experiment.  Run:
;;
;;   guix shell -m manifest.scm
;;
;; to enter the environment, possibly adding '--container' to avoid
;; interference with the host system.

(specifications->manifest
 '("coreutils"
   "findutils"
   "sed"
   "grep"

   "python-wrapper"
   "python-matplotlib"
   "python-numpy"
   "python-pandas"
   "python-scipy"
   "python-tqdm"))
