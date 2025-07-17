#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_27_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<27> mm_instantiator;

public:
  UnalignedM_27_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_27_MMInstantiatorWrapper__;

}  // namespace LLMMM
