#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_81_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<81> mm_instantiator;

public:
  UnalignedM_81_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_81_MMInstantiatorWrapper__;

}  // namespace LLMMM
