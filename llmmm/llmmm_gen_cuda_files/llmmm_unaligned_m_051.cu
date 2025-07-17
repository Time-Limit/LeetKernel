#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_51_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<51> mm_instantiator;

public:
  UnalignedM_51_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_51_MMInstantiatorWrapper__;

}  // namespace LLMMM
